"""
Evaluation Framework for Phase 3 Models

Supports multiple evaluation metrics for TSR and Figure Captioning models.
"""

import json
import time
import os
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd


class TSRMetrics:
    """Metrics for Table Structure Recognition evaluation."""
    
    @staticmethod
    def calculate_accuracy(predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
        """Calculate structure accuracy."""
        # Compare headers
        pred_headers = set(str(h) for h in predicted.get("headers", []))
        gt_headers = set(str(h) for h in ground_truth.get("headers", []))
        
        # Compare rows
        pred_rows = [tuple(str(c) for c in row) for row in predicted.get("rows", [])]
        gt_rows = [tuple(str(c) for c in row) for row in ground_truth.get("rows", [])]
        
        # Calculate metrics
        header_accuracy = len(pred_headers & gt_headers) / len(gt_headers) if gt_headers else 0.0
        row_accuracy = len(set(pred_rows) & set(gt_rows)) / len(gt_rows) if gt_rows else 0.0
        
        return (header_accuracy + row_accuracy) / 2.0
    
    @staticmethod
    def calculate_cell_accuracy(predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
        """Calculate cell-level accuracy."""
        pred_rows = predicted.get("rows", [])
        gt_rows = ground_truth.get("rows", [])
        
        if not pred_rows or not gt_rows:
            return 0.0
        
        total_cells = 0
        correct_cells = 0
        
        min_rows = min(len(pred_rows), len(gt_rows))
        for i in range(min_rows):
            pred_row = pred_rows[i]
            gt_row = gt_rows[i]
            min_cols = min(len(pred_row), len(gt_row))
            
            for j in range(min_cols):
                total_cells += 1
                if str(pred_row[j]).strip().lower() == str(gt_row[j]).strip().lower():
                    correct_cells += 1
        
        return correct_cells / total_cells if total_cells > 0 else 0.0
    
    @staticmethod
    def calculate_edit_distance(predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
        """Calculate edit distance between predicted and ground truth."""
        # Simple character-level edit distance
        pred_str = json.dumps(predicted, sort_keys=True)
        gt_str = json.dumps(ground_truth, sort_keys=True)
        
        # Levenshtein distance (simplified)
        def levenshtein(s1, s2):
            if len(s1) < len(s2):
                return levenshtein(s2, s1)
            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        distance = levenshtein(pred_str, gt_str)
        max_len = max(len(pred_str), len(gt_str))
        return 1.0 - (distance / max_len) if max_len > 0 else 0.0


class CaptionMetrics:
    """Metrics for Figure Captioning evaluation."""
    
    @staticmethod
    def calculate_bleu(predicted: str, ground_truth: str) -> float:
        """Calculate BLEU score."""
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            
            pred_tokens = predicted.lower().split()
            gt_tokens = ground_truth.lower().split()
            
            smoothing = SmoothingFunction().method1
            score = sentence_bleu([gt_tokens], pred_tokens, smoothing_function=smoothing)
            return float(score)
        except ImportError:
            # Fallback: simple word overlap
            pred_words = set(predicted.lower().split())
            gt_words = set(ground_truth.lower().split())
            return len(pred_words & gt_words) / len(gt_words) if gt_words else 0.0
    
    @staticmethod
    def calculate_rouge_l(predicted: str, ground_truth: str) -> float:
        """Calculate ROUGE-L score."""
        try:
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            scores = scorer.score(ground_truth, predicted)
            return scores['rougeL'].fmeasure
        except ImportError:
            # Fallback: simple longest common subsequence
            def lcs_length(s1, s2):
                m, n = len(s1), len(s2)
                dp = [[0] * (n + 1) for _ in range(m + 1)]
                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        if s1[i-1] == s2[j-1]:
                            dp[i][j] = dp[i-1][j-1] + 1
                        else:
                            dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                return dp[m][n]
            
            pred_words = predicted.lower().split()
            gt_words = ground_truth.lower().split()
            lcs = lcs_length(pred_words, gt_words)
            return lcs / len(gt_words) if gt_words else 0.0
    
    @staticmethod
    def calculate_meteor(predicted: str, ground_truth: str) -> float:
        """Calculate METEOR score."""
        try:
            from nltk.translate.meteor_score import meteor_score
            
            return float(meteor_score([ground_truth], predicted))
        except ImportError:
            # Fallback: word overlap with synonyms (simplified)
            pred_words = set(predicted.lower().split())
            gt_words = set(ground_truth.lower().split())
            overlap = len(pred_words & gt_words)
            return overlap / (len(pred_words) + len(gt_words) - overlap) if (pred_words or gt_words) else 0.0
    
    @staticmethod
    def calculate_cider(predicted: str, ground_truth: str) -> float:
        """Calculate CIDEr score (simplified)."""
        # Simplified CIDEr - full implementation requires sentence embeddings
        pred_words = predicted.lower().split()
        gt_words = ground_truth.lower().split()
        
        # Term frequency
        pred_tf = {word: pred_words.count(word) / len(pred_words) for word in set(pred_words)}
        gt_tf = {word: gt_words.count(word) / len(gt_words) for word in set(gt_words)}
        
        # Cosine similarity
        all_words = set(pred_words) | set(gt_words)
        dot_product = sum(pred_tf.get(w, 0) * gt_tf.get(w, 0) for w in all_words)
        pred_norm = sum(v**2 for v in pred_tf.values()) ** 0.5
        gt_norm = sum(v**2 for v in gt_tf.values()) ** 0.5
        
        return dot_product / (pred_norm * gt_norm) if (pred_norm * gt_norm) > 0 else 0.0
    
    @staticmethod
    def calculate_semantic_similarity(predicted: str, ground_truth: str) -> float:
        """Calculate semantic similarity using embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            model = SentenceTransformer('all-MiniLM-L6-v2')
            emb1 = model.encode(predicted)
            emb2 = model.encode(ground_truth)
            
            # Cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(similarity)
        except ImportError:
            # Fallback: simple word overlap
            pred_words = set(predicted.lower().split())
            gt_words = set(ground_truth.lower().split())
            return len(pred_words & gt_words) / len(gt_words | pred_words) if (pred_words or gt_words) else 0.0


class ModelEvaluator:
    """Evaluates models on test dataset."""
    
    def __init__(self, test_data_path: str):
        """
        Initialize evaluator.
        
        Args:
            test_data_path: Path to test dataset JSON file
        """
        self.test_data_path = Path(test_data_path)
        self.load_test_data()
    
    def load_test_data(self):
        """Load test dataset."""
        with open(self.test_data_path, 'r') as f:
            self.test_data = json.load(f)
    
    def evaluate_tsr_model(self, processor, model_name: str) -> Dict[str, Any]:
        """Evaluate TSR model."""
        results = []
        total_time = 0.0
        
        for item in self.test_data.get("tables", []):
            image_path = item["image_path"]
            ground_truth = item["ground_truth"]
            
            # Load image
            import cv2
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            # Process
            start_time = time.time()
            try:
                predicted = processor.process(image)
                inference_time = time.time() - start_time
                total_time += inference_time
                
                # Calculate metrics
                metrics = {
                    "structure_accuracy": TSRMetrics.calculate_accuracy(
                        predicted.get("structured_data", {}), ground_truth
                    ),
                    "cell_accuracy": TSRMetrics.calculate_cell_accuracy(
                        predicted.get("structured_data", {}), ground_truth
                    ),
                    "edit_distance": TSRMetrics.calculate_edit_distance(
                        predicted.get("structured_data", {}), ground_truth
                    ),
                    "inference_time": inference_time
                }
                results.append(metrics)
            except Exception as e:
                results.append({
                    "error": str(e),
                    "inference_time": time.time() - start_time
                })
        
        # Aggregate results
        if results:
            avg_metrics = {
                "model": model_name,
                "structure_accuracy": np.mean([r.get("structure_accuracy", 0) for r in results if "error" not in r]),
                "cell_accuracy": np.mean([r.get("cell_accuracy", 0) for r in results if "error" not in r]),
                "edit_distance": np.mean([r.get("edit_distance", 0) for r in results if "error" not in r]),
                "avg_inference_time": total_time / len(results),
                "total_samples": len(results),
                "successful_samples": len([r for r in results if "error" not in r])
            }
            return avg_metrics
        else:
            return {"model": model_name, "error": "No valid results"}
    
    def evaluate_caption_model(self, processor, model_name: str) -> Dict[str, Any]:
        """Evaluate captioning model."""
        results = []
        total_time = 0.0
        
        for item in self.test_data.get("figures", []):
            image_path = item["image_path"]
            ground_truth = item["ground_truth"]
            
            # Load image
            import cv2
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            # Process
            start_time = time.time()
            try:
                predicted_result = processor.process(image)
                inference_time = time.time() - start_time
                total_time += inference_time
                
                predicted_caption = predicted_result.get("caption", "")
                
                # Calculate metrics
                metrics = {
                    "bleu": CaptionMetrics.calculate_bleu(predicted_caption, ground_truth),
                    "rouge_l": CaptionMetrics.calculate_rouge_l(predicted_caption, ground_truth),
                    "meteor": CaptionMetrics.calculate_meteor(predicted_caption, ground_truth),
                    "cider": CaptionMetrics.calculate_cider(predicted_caption, ground_truth),
                    "semantic_similarity": CaptionMetrics.calculate_semantic_similarity(predicted_caption, ground_truth),
                    "inference_time": inference_time
                }
                results.append(metrics)
            except Exception as e:
                results.append({
                    "error": str(e),
                    "inference_time": time.time() - start_time
                })
        
        # Aggregate results
        if results:
            avg_metrics = {
                "model": model_name,
                "bleu": np.mean([r.get("bleu", 0) for r in results if "error" not in r]),
                "rouge_l": np.mean([r.get("rouge_l", 0) for r in results if "error" not in r]),
                "meteor": np.mean([r.get("meteor", 0) for r in results if "error" not in r]),
                "cider": np.mean([r.get("cider", 0) for r in results if "error" not in r]),
                "semantic_similarity": np.mean([r.get("semantic_similarity", 0) for r in results if "error" not in r]),
                "avg_inference_time": total_time / len(results),
                "total_samples": len(results),
                "successful_samples": len([r for r in results if "error" not in r])
            }
            return avg_metrics
        else:
            return {"model": model_name, "error": "No valid results"}

