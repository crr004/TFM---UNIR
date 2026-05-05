import json
from pathlib import Path

import numpy as np
import shap
import torch
from captum.attr import LayerIntegratedGradients
from lime.lime_text import LimeTextExplainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


class BetoExplainer:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.labels = ["REAL", "FAKE"]
        self.clf_pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            top_k=None,
            device=0 if self.device.type == "cuda" else -1,
        )

    @staticmethod
    def _score_map_from_pipeline_sample(sample):
        # Expected (newer API): list[dict(label, score)]
        if isinstance(sample, list):
            return {item["label"].upper(): float(item["score"]) for item in sample}

        # Fallback (older/single-label output): dict(label, score)
        if isinstance(sample, dict) and "label" in sample and "score" in sample:
            label = str(sample["label"]).upper()
            score = float(sample["score"])
            if label in {"FAKE", "LABEL_1"}:
                return {"FAKE": score, "REAL": 1.0 - score}
            return {"REAL": score, "FAKE": 1.0 - score}

        return {}

    def predict_proba(self, texts):
        outputs = self.clf_pipeline(list(texts), truncation=True, max_length=256)
        probs = []
        for sample in outputs:
            score_map = self._score_map_from_pipeline_sample(sample)
            p_real = score_map.get("REAL", score_map.get("LABEL_0", 0.0))
            p_fake = score_map.get("FAKE", score_map.get("LABEL_1", 0.0))
            probs.append([p_real, p_fake])
        return np.array(probs)

    def _predict_with_details(self, text: str):
        probs = self.predict_proba([text])[0]
        pred_class = int(np.argmax(probs))
        pred_prob = float(probs[pred_class])
        return probs, pred_class, pred_prob

    @staticmethod
    def _normalize_word(word: str) -> str:
        return "".join(
            ch
            for ch in word.lower()
            if ch.isalnum() or ch in {"_", "-", "á", "é", "í", "ó", "ú", "ñ", "ü"}
        )

    def _word_occlusion_importance(self, text: str, target_class: int):
        words = text.split()
        if not words:
            return [], []

        base_prob = float(self.predict_proba([text])[0][target_class])
        ranked = []

        for idx in range(len(words)):
            perturbed_words = words[:idx] + words[idx + 1 :]
            perturbed_text = " ".join(perturbed_words).strip()

            if not perturbed_text:
                perturbed_prob = 0.0
            else:
                perturbed_prob = float(
                    self.predict_proba([perturbed_text])[0][target_class]
                )

            importance = base_prob - perturbed_prob
            ranked.append(
                {
                    "index": idx,
                    "word": words[idx],
                    "importance": float(importance),
                    "prob_without_word": float(perturbed_prob),
                }
            )

        ranked = sorted(ranked, key=lambda x: x["importance"], reverse=True)
        return words, ranked

    @staticmethod
    def _build_text_from_indices(words, selected_indices, keep_selected=True):
        selected = set(selected_indices)
        if keep_selected:
            out = [w for i, w in enumerate(words) if i in selected]
        else:
            out = [w for i, w in enumerate(words) if i not in selected]
        return " ".join(out).strip()

    @staticmethod
    def _auc(points):
        if len(points) < 2:
            return None
        points = sorted(points, key=lambda x: x["fraction"])
        x = np.array([p["fraction"] for p in points], dtype=float)
        y = np.array([p["prob"] for p in points], dtype=float)
        if hasattr(np, "trapezoid"):
            return float(np.trapezoid(y, x))
        return float(np.trapz(y, x))

    def explain_lime(self, text: str, output_path: Path, num_features: int = 12):
        explainer = LimeTextExplainer(class_names=self.labels)
        exp = explainer.explain_instance(
            text,
            classifier_fn=self.predict_proba,
            labels=[0, 1],
            num_features=num_features,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(exp.as_html())

        return {
            "path": str(output_path),
            "top_real": exp.as_list(label=0),
            "top_fake": exp.as_list(label=1),
        }

    # VERSIÓN ANTERIOR
    # Esta versión funcionaba con textos cortos, pero podía fallar con textos largos
    # porque SHAP llamaba internamente al pipeline sin truncation=True ni max_length=256.
    #
    # def explain_shap(self, texts, output_path: Path, max_samples: int = 8):
    #     texts = list(texts)[:max_samples]
    #     explainer = shap.Explainer(self.clf_pipeline)
    #     shap_values = explainer(texts)
    #
    #     records = []
    #     for i, text in enumerate(texts):
    #         sample_values = shap_values.values[i]
    #         sample_data = shap_values.data[i]
    #         probs = self.predict_proba([text])[0]
    #         pred_class = int(np.argmax(probs))
    #
    #         token_scores = []
    #         for token, scores in zip(sample_data, sample_values):
    #             token_scores.append(
    #                 {
    #                     "token": str(token),
    #                     "score": float(scores[pred_class]),
    #                 }
    #             )
    #
    #         token_scores = sorted(
    #             token_scores, key=lambda x: abs(x["score"]), reverse=True
    #         )
    #         records.append(
    #             {
    #                 "text": text,
    #                 "predicted_class": self.labels[pred_class],
    #                 "top_tokens": token_scores[:20],
    #             }
    #         )
    #
    #     output_path = Path(output_path)
    #     output_path.parent.mkdir(parents=True, exist_ok=True)
    #     with open(output_path, "w", encoding="utf-8") as f:
    #         json.dump(records, f, indent=2, ensure_ascii=False)
    #
    #     return {"path": str(output_path), "samples": len(records)}

    def explain_shap(self, texts, output_path: Path, max_samples: int = 8):
        texts = list(texts)[:max_samples]

        # SHAP puede llamar internamente al pipeline sin truncation=True.
        # Por eso truncamos y reconstruimos los textos antes de pasarlos a SHAP,
        # respetando el mismo max_length usado durante entrenamiento/evaluación.
        truncated_texts = []
        for text in texts:
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=256,
                return_tensors=None,
            )
            truncated_text = self.tokenizer.decode(
                encoded["input_ids"],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            truncated_texts.append(truncated_text)

        explainer = shap.Explainer(self.clf_pipeline)
        shap_values = explainer(truncated_texts)

        records = []
        for i, text in enumerate(truncated_texts):
            sample_values = shap_values.values[i]
            sample_data = shap_values.data[i]
            probs = self.predict_proba([text])[0]
            pred_class = int(np.argmax(probs))

            token_scores = []
            for token, scores in zip(sample_data, sample_values):
                token_scores.append(
                    {
                        "token": str(token),
                        "score": float(scores[pred_class]),
                    }
                )

            token_scores = sorted(
                token_scores, key=lambda x: abs(x["score"]), reverse=True
            )

            records.append(
                {
                    "text": text,
                    "predicted_class": self.labels[pred_class],
                    "top_tokens": token_scores[:20],
                }
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        return {"path": str(output_path), "samples": len(records)}

    def explain_integrated_gradients(
        self, text: str, output_path: Path, target: int | None = None
    ):
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
            pred = int(torch.argmax(logits, dim=1).item())

        target = pred if target is None else int(target)

        def forward_func(ids, mask):
            out = self.model(input_ids=ids, attention_mask=mask).logits
            return out

        lig = LayerIntegratedGradients(forward_func, self.model.get_input_embeddings())
        attributions, delta = lig.attribute(
            inputs=input_ids,
            additional_forward_args=(attention_mask,),
            target=target,
            n_steps=32,
            return_convergence_delta=True,
        )

        token_attr = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())

        token_scores = [
            {"token": tok, "score": float(score)}
            for tok, score in zip(tokens, token_attr)
            if tok not in {"[CLS]", "[SEP]", "[PAD]"}
        ]
        token_scores = sorted(token_scores, key=lambda x: abs(x["score"]), reverse=True)

        output = {
            "predicted_class": self.labels[pred],
            "target_class": self.labels[target],
            "convergence_delta": float(torch.mean(torch.abs(delta)).item()),
            "top_tokens": token_scores[:25],
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        return {"path": str(output_path), "predicted_class": self.labels[pred]}

    def explain_faithfulness(
        self,
        text: str,
        output_path: Path,
        fractions=(0.1, 0.2, 0.3, 0.4, 0.5),
    ):
        probs, pred_class, base_prob = self._predict_with_details(text)
        words, ranked = self._word_occlusion_importance(text, target_class=pred_class)

        if not words:
            output = {
                "predicted_class": self.labels[pred_class],
                "base_prob": base_prob,
                "error": "Texto vacío tras tokenización por espacios.",
            }
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            return {
                "path": str(output_path),
                "predicted_class": self.labels[pred_class],
                "mean_comprehensiveness": float("nan"),
                "mean_sufficiency": float("nan"),
                "deletion_auc": float("nan"),
                "insertion_auc": float("nan"),
                "n_words": 0,
                "base_pred_prob": float(base_prob),
            }

        top_indices = [item["index"] for item in ranked]
        n_words = len(words)

        comp_points = []
        suff_points = []
        deletion_curve = []
        insertion_curve = []

        for frac in fractions:
            k = max(1, int(round(frac * n_words)))
            selected = top_indices[:k]

            comp_text = self._build_text_from_indices(
                words, selected, keep_selected=False
            )
            suff_text = self._build_text_from_indices(
                words, selected, keep_selected=True
            )

            comp_prob = (
                0.0
                if not comp_text
                else float(self.predict_proba([comp_text])[0][pred_class])
            )
            suff_prob = (
                0.0
                if not suff_text
                else float(self.predict_proba([suff_text])[0][pred_class])
            )

            comprehensiveness = base_prob - comp_prob
            sufficiency = base_prob - suff_prob

            point = {
                "fraction": float(frac),
                "k": int(k),
                "prob": float(comp_prob),
                "delta": float(comprehensiveness),
            }
            comp_points.append(point)
            deletion_curve.append(point)

            point = {
                "fraction": float(frac),
                "k": int(k),
                "prob": float(suff_prob),
                "delta": float(sufficiency),
            }
            suff_points.append(point)
            insertion_curve.append(point)

        output = {
            "predicted_class": self.labels[pred_class],
            "base_probs": {
                "REAL": float(probs[0]),
                "FAKE": float(probs[1]),
            },
            "base_pred_prob": float(base_prob),
            "n_words": int(n_words),
            "top_words": ranked[:25],
            "comprehensiveness": {
                "points": comp_points,
                "mean_delta": float(np.mean([p["delta"] for p in comp_points])),
            },
            "sufficiency": {
                "points": suff_points,
                "mean_delta": float(np.mean([p["delta"] for p in suff_points])),
            },
            "deletion_curve": {
                "points": deletion_curve,
                "auc": self._auc(deletion_curve),
            },
            "insertion_curve": {
                "points": insertion_curve,
                "auc": self._auc(insertion_curve),
            },
            "notes": {
                "comprehensiveness": "Mayor delta indica mayor dependencia del modelo en palabras top.",
                "sufficiency": "Delta cercano a 0 indica que las palabras top retienen suficiente evidencia.",
            },
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        return {
            "path": str(output_path),
            "predicted_class": self.labels[pred_class],
            "mean_comprehensiveness": output["comprehensiveness"]["mean_delta"],
            "mean_sufficiency": output["sufficiency"]["mean_delta"],
            "deletion_auc": output["deletion_curve"]["auc"],
            "insertion_auc": output["insertion_curve"]["auc"],
            "n_words": int(n_words),
            "base_pred_prob": float(base_prob),
        }
