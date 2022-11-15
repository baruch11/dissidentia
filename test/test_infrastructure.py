"""Unit tests for infrastucture"""
from dissidentia.infrastructure.grand_debat import GDAnswers


def test_import():
    """test data importation"""
    GDAnswers().import_data()


def test_load():
    """test load 10 documents"""
    gdans = GDAnswers()
    print()
    print(f"Selected quesion(s): {gdans.QUESTIONS}")
    ans = gdans.load_data(10)
    print("10 samples:")
    print(ans)
