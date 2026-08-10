import json
import re


INPUT = "outputs/generation_eval/standard_lora_seed11_gsm8k_20.json"
OUTPUT = "outputs/generation_eval/standard_lora_seed11_gsm8k_20_rescored.json"


def extract_final_number(text):
    if "Final Answer:" in text:
        candidate = text.split("Final Answer:", 1)[1]
        matches = re.findall(r"-?\$?\d[\d,]*(?:\.\d+)?", candidate)

        if not matches:
            return None

        value = matches[0]
    else:
        matches = re.findall(r"-?\$?\d[\d,]*(?:\.\d+)?", text)

        if not matches:
            return None

        value = matches[-1]

    return value.replace("$", "").replace(",", "").strip()


def normalize_number(value):
    if value is None:
        return None

    try:
        number = float(value)

        if number.is_integer():
            return str(int(number))

        return f"{number:.10f}".rstrip("0").rstrip(".")

    except ValueError:
        return value.strip()


with open(INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)


correct = 0

for row in data["results"]:

    pred = normalize_number(
        extract_final_number(row["generated_text"])
    )

    gold = normalize_number(
        extract_final_number(row["gold_answer"])
    )

    is_correct = (
        pred is not None
        and gold is not None
        and pred == gold
    )

    row["predicted_number_rescored"] = pred
    row["gold_number_rescored"] = gold
    row["correct_rescored"] = is_correct

    if is_correct:
        correct += 1

    print(
        f"{row['index'] + 1:02d}: "
        f"pred={pred} "
        f"gold={gold} "
        f"correct={is_correct}"
    )


total = len(data["results"])
accuracy = correct / total if total else 0.0

data["rescored_correct"] = correct
data["rescored_accuracy"] = accuracy


with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)


print()
print("=" * 50)
print("RESCORED GSM8K RESULTS")
print("=" * 50)
print(f"Questions : {total}")
print(f"Correct   : {correct}")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Percent   : {accuracy * 100:.2f}%")
print(f"Saved to  : {OUTPUT}")