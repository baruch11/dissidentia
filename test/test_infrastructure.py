"""Unit tests for infrastucture"""
from dissidentia.infrastructure.grand_debat import GDAnswers
from dissidentia.infrastructure.sentencizer import sentencize

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


def test_sentencize():
    """Check the number of sentences found after split"""
    test_text = [
        ("Salut ! Comment vas-tu ?", 2),
        ("J'ai oublié l'espace en fin de phrase.Mais c'est pas grave", 2),
        ("Je m'amuse à mettre plein de points.......", 1),
        ("J'ai un score de 0.56.C'est pas mal", 2),
    ]

    for text, nb_sents in test_text:
        sents = sentencize(text)
        print(sents)
        assert len(sents) == nb_sents
