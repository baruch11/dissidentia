"""Unit tests for infrastucture"""
from dissidentia.infrastructure.grand_debat import GDAnswers
from dissidentia.infrastructure.sentencizer import sentencize
from dissidentia.infrastructure.doccano import DoccanoDataset


def test_import_gd_answers():
    """test data importation from grand debat site"""
    GDAnswers().import_data()


def test_load_gd_answers():
    """test load 10 documents from grand debat"""
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


def test_doccano():
    """ test load train test from doccano """
    dds = DoccanoDataset()
    df = dds.load_data()
    print(f"labels:\n{df.label.value_counts()}")
    assert all(col in df.columns
               for col in ["label", "text", "question_id", "sent_id"])
