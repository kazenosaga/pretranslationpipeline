import openai, anthropic
import argparse, csv
import re, os, sys
import gloss_retriever as gr

client = openai.OpenAI(
    api_key="******************************************", 
    base_url="https://api.aitunnel.ru/v1/"
)

# Параметры
target_lang = "English"
VERBOSITY = 3                                   # Уровень вывода отладки, 4 -- максимальный
EMBED_INDEX_DIR = "./resources/gloss_index"     # папка с эмбеддингом
TOP_K = 8
MIN_SIM = 0.25
TEMPERATURE = 0.2
CONTEXT_BEFORE = 15
CONTEXT_AFTER = 15
USE_CLAUDIE = 0                                 # ### Только для Claude напрямую, мы это не используем
USE_AITUNNEL = 1
API_TIMEOUT = 60


# TEXT_TO_TRANSLATE = "Отличия от курса по версии 12. В курсе используется PostgreSQL версии 16. Учтены новые возможности, появившиеся в версиях 13, 14, 15 и 16. Материал некоторых тем переработан, хотя в целом массив информации курса по сравнению с версией 12 не претерпел существенных изменений. Преподавателю необходимо изучить документ «Изменения в материалах курса» (https://edu.postgrespro.ru/16/dev1-16/dev1_new_features.pdf). В нем содержится информация о наиболее важных изменениях, появившихся в версиях 13, 14, 15 и 16 и нашедших отражение в курсе."
TEXT_TO_TRANSLATE = "Тема знакомит с логическими резервными копиями отдельных элементов: таблиа (COPY), база данных (pg_dump) и весь кластер (pg_dumpall). Еще хотелось бы посмотреть, что такое Методика хранения сверхбольших атрибутов."

SYSTEM_PROMPT = (
    "You are a professional technical translator specializing in databases and PostgreSQL. "
    "You work with presentation files and scripts that may include code, diagram captions, and technical text.\n\n"
    "TRANSLATION APPROACH:\n"
    "- First understand the complete meaning and intent of the segment within its context\n"
    "- Translate ideas and concepts, not individual words\n"
    "- Produce natural, fluent translations that sound like original {target_lang} text\n"
    "- Rephrase sentence structures when necessary for better flow and readability\n"
    "- Maintain technical accuracy while improving linguistic naturalness\n"
    "- Always check for PostgreSLQ-specific terminology and pay attention to possible glossary hits\n"
    "- Never perform literal, word-for-word translation\n\n"
    "OUTPUT CONSTRAINTS:\n"
    "1) Return ONLY the final translation text, no notes or explanations\n"
    "2) Preserve all formatting, code, URLs, numbers, and technical symbols exactly\n"
    "3) Maintain HTML/markup structure if present\n"
    "4) Never add commentary, rationale, or thinking process\n"
    "5) If text is code, filenames, or doesn't need translation, return it unchanged\n\n"
    "GLOSSARY POLICY:\n"
    "- Strictly use provided glossary terms when they appear in the source\n"
    "- Do not invent alternatives for glossary entries\n"
    "- Apply natural capitalization within sentence context\n\n"
    "CONTEXT USAGE:\n"
    "- Use surrounding segments to understand the broader topic and maintain consistency\n"
    "- Reference previous translations for terminology and style consistency\n"
    "- If existing translation is provided, preserve it unless it contains clear errors\n"
    "- Ensure the current translation flows naturally with the surrounding content\n"
    "THINGS TO AVOID AT ALL COSTS:\n"
    "- Unnatural-sounding or verbatum translations. Always double-check and rewrite for a natural flow.\n"
    "- Broken tags. Always leave tags intact and in their place.\n"
    "- Reasoning leftovers, previous attempts, revisions, explainations of your choice of translation. Always return just the translated line or lines, nothing else.\n"
    "ADDITIONAL REQUIREMENTS:\n"
    "- Use the – instead of —. Always surround it by spaces.\n"
    "- When the Russian text says В 19-й версии and speaks about PostgreSQL, translate it as In PostgreSQL 19, not as In version 19.\n"
    
    
    # Ручной глоссарий после отбора идет сюда
    # Embedded глоссарий идет сюда
    # Контекст идет сюда
    # USER_PROMPT идет сюда
    # Текст на перевод идет сюда
)

USER_PROMPT = (
    "\n\nTranslate the following segment into English. Then, rephrase your first attempt at translation to make it sound more natural. Return only that revised translation, discard the original attempt. Never include any reasoning, previous attempts, alternative translations or your internal thinking into the output. The output is just the translation, nothing else.\n\n"
)

### Ручной глоссарий

def load_manual_glossary(path="./resources/gloss.csv"):
    if not os.path.exists(path):
        if VERBOSITY >= 2:
            print ("!!! No manual glossary found at: " + path)
        return []
    with open(path, encoding="utf-8-sig") as f:
        return [(r[0].strip(), r[1].strip()) for r in csv.reader(f) if len(r) >= 2 and r[0].strip() and r[1].strip()]

GLOSSARY_MANUAL = load_manual_glossary()

#GLOSSARY_MANUAL = [
#    ("PostgreSQL", "PostgreSQL"),
#    ("материалы курса", "course materials"),
#    ("преподаватель", "instructor"),
#]

### Отбор релевантных терминов из ручного глоссария

def select_glossary_entries_manual(text: str):
    matches = []
    for src, tgt in GLOSSARY_MANUAL:
        if " " in src.strip():
            pattern = r"(?i)(?<!\w)" + re.escape(src.strip()) + r"(?!\w)"
        else:
            pattern = r"(?i)\b" + re.escape(src.strip()) + r"\b"
        if re.search(pattern, text):
            matches.append((src, tgt))  
    return matches

### Сбор контекста из предыдущих и следующих строк CSV

def build_context_prompt(rows, current_index, src_col, context_before=7, context_after=3):
    context_lines = []
    context_lines.append("\n\n(Context for reference:)")

    # Собрать строки на контекст в кол-ве context_before+1+context_after
    start_idx = max(0, current_index - context_before)
    end_idx = min(current_index + context_after, len(rows)-1)
    for i in range(start_idx, end_idx):
        src = (rows[i].get(src_col) or "").strip()
        tgt = (rows[i].get("en") or "").strip()
        if src:  # Не пустой
            if tgt:
                context_lines.append(src +" => " + tgt)
            else:
                context_lines.append(src)
    context_lines.append("(End of context)")

    return "\n".join(context_lines)

### СБОР ЗАПРОСА ИЗ КУСОЧКОВ

def translate (text: str, context = "") -> str:
    
    if VERBOSITY >= 1:
        print (">>> Text to translate:\n" + text +"\n")

    ### + ручной глоссарий

    relevant = select_glossary_entries_manual(text)
    if relevant:
        glossary_block = "\nGlossary: " + "\n".join([f"- {src} => {tgt}" for src, tgt in relevant]) + "\n"
    else:
        glossary_block = ""

    if (VERBOSITY >= 3): 
        print (">>> Glossary matches, manual: " + str(relevant)+"\n")

    ### + embedded глоссарий

    gr.init(EMBED_INDEX_DIR)

    if VERBOSITY >= 3:
        print (f">>> Embedding index loaded: {gr.have_index()} ({EMBED_INDEX_DIR})\n")

    if gr.have_index():
        relevant = gr.retrieve(text, top_k=TOP_K, min_sim=MIN_SIM)
        glossary_block = "\nPossible relevant terms:\n" + "\n".join([f"- {src} => {tgt}" for src, tgt in relevant]) + "\n"

    if (VERBOSITY >= 2): 
        print (">>> Glossary matches, embedded: " + str(relevant)+"\n")

    ### Обращение к серверу с LLM
    if USE_CLAUDIE: ### Только для Claude напрямую, мы это не используем
        client = anthropic.Anthropic(api_key="*********************************************************")
        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens = 1024,
            temperature=TEMPERATURE,
            timeout = API_TIMEOUT,
            messages=[
                {
                    "role": "user",
                    "content": SYSTEM_PROMPT.format(target_lang=target_lang) + glossary_block + context + USER_PROMPT + text,
                }
            ],
        )
    else:
        client = openai.OpenAI(
        api_key="*********************************************************",
        base_url="https://api.aitunnel.ru/v1/"
        )
        response = client.chat.completions.create(
            model="claude-sonnet-5",
            temperature=TEMPERATURE,
            max_tokens = 2048,
            timeout = API_TIMEOUT,
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT.format(target_lang=target_lang),
                            "cache_control": {
                                "type": "ephemeral"
                            }
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": glossary_block + context + USER_PROMPT + text,

                    
                },
            ],
        )    

    if (VERBOSITY >= 4): 
        print ("=== FULL PROMPT WITH CONTEXT ===")
        print (SYSTEM_PROMPT.format(target_lang=target_lang) + glossary_block + context + USER_PROMPT + text)
        print ("=== END OF FULL PROMPT WITH CONTEXT ===\n\n")

    ### Конец обращения

    #def sanitize_output(s: str) -> str:
    #    if VERBOSITY >=3:
    #        return (s or "").replace("<think>", "").replace("</think>", "").strip()
    #    if VERBOSITY >=1:
    #        return re.sub(r'<think>.*?</think>\s*', '', s, flags=re.DOTALL)



    if USE_CLAUDIE:
        #return response.message.content
        text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
        return text
    

    
    else:
        return response.choices[0].message.content
### Конец запроса

def pick_from_message (response_text :str, resposne_type: str):
    if USE_CLAUDIE or USE_AITUNNEL:
        if (VERBOSITY >= 1):
            print (">>> Translation:\n" + response_text)
        return response_text
    
    ### Отделение ризонинга и вывода
    think_block = re.search(r'<think>([\s\S]*?)</think>', response_text)
    response_reasoning = think_block.group(1).strip()
    resposne_translation = re.sub(r'<think>([\s\S]*?)</think>', '', response_text).strip()

    #print(sanitize_output(response.choices[0].message.content))

    if resposne_type == "reasoning":
        if (VERBOSITY >= 4): 
            print (">>> Reasoning:\n", response_reasoning, "\n")
        return response_reasoning
    
    else:
        if (VERBOSITY >= 1): 
            print (">>> Translation:\n", resposne_translation, "\n")
        return resposne_translation
    

### Парсер параметров
def parse_args():
    ap = argparse.ArgumentParser(description="Translate a string or a CSV.")
    ap.add_argument("-s", "--string", help="Translate a single string.")
    ap.add_argument("-f", "--file", help="Translate a CSV file. Read column 1, write to column 2.")
    ap.add_argument("-ow", "--overwrite", action="store_true", help="In a CSV file, overwrite the translation if exists.")
    ap.add_argument("-nc", "--nocontext", action="store_true", help="In a CSV file, do not send CONTEXT_BEFORE strings and CONTEXT_AFTER strings.")
    ap.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2, 3, 4], default=3, help="Set verbosity level: 0=none, 4=all")
    ap.add_argument("-lr", "--logreasoning", action="store_true", help="In a CSV file, log reasoning in the 3rd column.")
    return ap.parse_args()

def main():
    args = parse_args()
    
    global VERBOSITY
    VERBOSITY = args.verbosity # По умолчанию 3, всё кроме ризонинга
    global CONTEXT_BEFORE
    global CONTEXT_AFTER

    message = ""

    # Проверить, что не оба
    if args.string and args.file:
        print("Error: use either -s/--string or -f/--file, not both.", file=sys.stderr)
        sys.exit(2)

    # Строка
    if args.string:
        pick_from_message(translate(args.string), "translation")
        return

    # CSV с двумя колонками
    if args.file:
        in_csv = args.file
        if not (os.path.isfile(in_csv) and in_csv.lower().endswith(".csv")):
            raise SystemExit("The path passed to --file must be an existing .csv file.")
    
        # Читаем файл
        with open(in_csv, "r", encoding="utf-8-sig", newline="") as f_in:
            reader = csv.DictReader(f_in)
            fieldnames = list(reader.fieldnames or [])
            if not fieldnames:
                raise SystemExit("CSV has no header.")
            src_col = fieldnames[0]
            rows = list(reader)

        out_fields = fieldnames.copy()
        if "en" not in out_fields:
            out_fields.append("en")
        
        if (args.logreasoning & USE_CLAUDIE==0):
            out_fields = fieldnames.copy()
            if "reasoning" not in out_fields:
                out_fields.append("reasoning")

#        out_csv = os.path.splitext(in_csv)[0] + "(en).csv"
#
#        with open(out_csv, "w", encoding="utf-8", newline="") as f_out:
#            w = csv.DictWriter(f_out, fieldnames=out_fields)
#            w.writeheader()
#
        # Перевод и запись каждой строки

        for i, row in enumerate(rows, start=1):
            src = (row.get(src_col) or "").strip()
            existing_translation = row.get("en", "").strip()
        
            # Пропускаем, если есть существующий перевод
            if existing_translation and not args.overwrite:
                if VERBOSITY >= 1:
                    print(f">>> Skipping row {i}/{len(rows)} - already translated")
                continue
            
            # Если нет или стоит флаг -ow, то пишем как обычно
            if src and (not existing_translation or args.overwrite):
                # Если кириллицы нет, пишем сразу в таргет
                if not any('\u0400' <= char <= '\u04FF' for char in src):
                    if VERBOSITY >= 2:
                        print(f">>> No Cyrillic in row {i}/{len(rows)} - copying as-is")
                    row["en"] = src
                # Иначе работаем как дальше
                else:

                    context_prompt = build_context_prompt(rows, i, src_col, CONTEXT_BEFORE, CONTEXT_AFTER)

                    if VERBOSITY >= 4:
                        print(f"+++ CONTEXT START +++ \n")
                        print(context_prompt)
                        print(f"+++ CONTEXT END +++ \n")
                    message = translate(src, context_prompt)
                    row["en"] = pick_from_message(message, "translation")
                                    
                    if args.logreasoning:
                        row["reasoning"] = pick_from_message(message, "reasoning")

                # И сразу пишем
                with open(in_csv, "w", encoding="utf-8", newline="") as f_out:
                    w = csv.DictWriter(f_out, fieldnames=out_fields)
                    w.writeheader()
                    w.writerows(rows)
                
                if VERBOSITY >= 1:
                    print(f">>> Processed row {i}/{len(rows)}\n")
        
        return
    
    # Если без аргументов, переводим захардкоженный текст
    pick_from_message(translate(TEXT_TO_TRANSLATE), "translation")

if __name__ == "__main__":
    main()