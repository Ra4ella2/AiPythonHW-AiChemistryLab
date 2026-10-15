import csv
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rdkit import Chem

MODEL_NAME = "khanfs/ChemSolubilityBERTa"
OUTPUT_FILE = "chemistry_results.csv"

molecules = [
    {
        "name": "Етанол",
        "formula": "C2H6O",
        "smiles": "CCO"
    },
    {
        "name": "Оцтова кислота",
        "formula": "C2H4O2",
        "smiles": "CC(=O)O"
    },
    {
        "name": "Бензен",
        "formula": "C6H6",
        "smiles": "c1ccccc1"
    },
    {
        "name": "Метанол",
        "formula": "CH4O",
        "smiles": "CO"
    },
    {
        "name": "Ацетон",
        "formula": "C3H6O",
        "smiles": "CC(=O)C"
    },
    {
        "name": "Толуен",
        "formula": "C7H8",
        "smiles": "Cc1ccccc1"
    },
    {
        "name": "Фенол",
        "formula": "C6H6O",
        "smiles": "Oc1ccccc1"
    },
    {
        "name": "Анілін",
        "formula": "C6H7N",
        "smiles": "Nc1ccccc1"
    },
    {
        "name": "Пропанол",
        "formula": "C3H8O",
        "smiles": "CCCO"
    },
    {
        "name": "Етиленгліколь",
        "formula": "C2H6O2",
        "smiles": "OCCO"
    }
]

print("Завантаження моделі...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    print("Модель завантажено.")

except Exception as error:
    print("Помилка завантаження моделі:")
    print(error)
    exit()

def valid_smiles(smiles):
    try:
        molecule = Chem.MolFromSmiles(smiles)

        return molecule is not None

    except:
        return False

def predict_solubility(smiles):

    if not valid_smiles(smiles):
        raise ValueError("Некоректний SMILES.")

    inputs = tokenizer(
        smiles,
        return_tensors="pt",
        truncation=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    value = outputs.logits.squeeze().item()

    return value

def save_results(results):

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "molecule_name",
            "formula",
            "smiles",
            "predicted_property",
            "predicted_value"
        ])

        for item in results:

            writer.writerow([
                item["name"],
                item["formula"],
                item["smiles"],
                "aqueous_solubility_logS",
                round(item["value"], 4)
            ])

    print(
        f"\nРезультати збережено у {OUTPUT_FILE}"
    )

def analyze_one():

    print("\n===== ANALYZE MOLECULE =====")

    name = input("Назва речовини: ").strip()
    formula = input("Хімічна формула: ").strip()
    smiles = input("SMILES: ").strip()

    if not name or not formula or not smiles:
        print("Помилка: всі поля повинні бути заповнені.")
        return

    try:

        value = predict_solubility(smiles)

        print("\n===== RESULT =====")

        print("Речовина:", name)
        print("Формула:", formula)
        print("SMILES:", smiles)

        print(
            "Прогнозована властивість:",
            "водна розчинність (logS)"
        )

        print(
            "Прогнозоване значення:",
            round(value, 4),
            "log mol/L"
        )

    except Exception as error:

        print("Помилка аналізу:")
        print(error)

def analyze_all():

    results = []

    print("\n===== CHEMISTRY ANALYSIS =====")

    for molecule in molecules:

        try:

            value = predict_solubility(
                molecule["smiles"]
            )

            result = {
                "name": molecule["name"],
                "formula": molecule["formula"],
                "smiles": molecule["smiles"],
                "value": value
            }

            results.append(result)

            print(
                f"\n{molecule['name']}"
            )

            print(
                "Формула:",
                molecule["formula"]
            )

            print(
                "SMILES:",
                molecule["smiles"]
            )

            print(
                "Прогнозована розчинність:",
                round(value, 4),
                "log mol/L"
            )

        except Exception as error:

            print(
                f"\nПомилка для {molecule['name']}:"
            )

            print(error)

    save_results(results)

def show_results():

    try:

        with open(
            OUTPUT_FILE,
            "r",
            encoding="utf-8-sig"
        ) as file:

            print("\n===== SAVED RESULTS =====")

            print(file.read())

    except FileNotFoundError:

        print(
            "Файл результатів ще не створено."
        )

def search_molecule():

    name = input(
        "Введіть назву речовини: "
    ).lower()

    for molecule in molecules:

        if name in molecule["name"].lower():

            try:

                value = predict_solubility(
                    molecule["smiles"]
                )

                print("\n===== RESULT =====")

                print(
                    "Назва:",
                    molecule["name"]
                )

                print(
                    "Формула:",
                    molecule["formula"]
                )

                print(
                    "SMILES:",
                    molecule["smiles"]
                )

                print(
                    "Розчинність:",
                    round(value, 4),
                    "log mol/L"
                )

                return

            except Exception as error:

                print(error)
                return

    print("Речовину не знайдено.")

while True:

    print("""
================================
       AI CHEMISTRY LAB
================================

1. Проаналізувати одну сполуку
2. Проаналізувати 10 сполук
3. Показати результати
4. Пошук речовини
0. Вихід
""")

    choice = input("Оберіть пункт: ")

    if choice == "1":
        analyze_one()

    elif choice == "2":
        analyze_all()

    elif choice == "3":
        show_results()

    elif choice == "4":
        search_molecule()

    elif choice == "0":
        print("Програму завершено.")
        break

    else:
        print("Невірний пункт меню.")