"""Confere a estrutura do dataset exigida pelo enunciado (32/4/4 por classe, rótulos YOLO válidos)."""
import json
import unittest
from pathlib import Path

DATASET = Path(__file__).resolve().parent.parent / "data" / "yolo_dataset"
ESPERADO = {"train": 32, "val": 4, "test": 4}
CLASSES = {"vaca": 0, "caminhao": 1}


class DatasetTest(unittest.TestCase):
    def test_proporcao_por_classe(self):
        for split, quantidade in ESPERADO.items():
            for classe in CLASSES:
                imagens = list((DATASET / "images" / split).glob(f"{classe}_*.jpg"))
                self.assertEqual(len(imagens), quantidade, f"{split}/{classe}")

    def test_rotulos_yolo_validos(self):
        for split in ESPERADO:
            for imagem in (DATASET / "images" / split).glob("*.jpg"):
                rotulo = DATASET / "labels" / split / (imagem.stem + ".txt")
                self.assertTrue(rotulo.exists(), rotulo.name)
                linhas = [l.split() for l in rotulo.read_text().splitlines() if l.strip()]
                self.assertGreaterEqual(len(linhas), 1)
                for linha in linhas:
                    self.assertEqual(len(linha), 5)
                    self.assertEqual(int(linha[0]), CLASSES[imagem.stem.split("_")[0]])
                    self.assertTrue(all(0.0 <= float(v) <= 1.0 for v in linha[1:]))

    def test_manifesto_cobre_todas_as_imagens(self):
        manifesto = json.loads((DATASET / "manifest.json").read_text())
        self.assertEqual(len(manifesto), 80)
        for item in manifesto:
            self.assertTrue((DATASET / item["arquivo"]).exists())


if __name__ == "__main__":
    unittest.main()
