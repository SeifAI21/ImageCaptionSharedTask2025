"""
LLM-as-a-Judge evaluation for Arabic captions
"""

import re
from typing import Optional
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm
import pandas as pd
import evaluation_config as config

class Judgement(BaseModel):
    """Pydantic model for structured LLM judge output."""
    score: int


class LLMJudge:
    """LLM-based evaluation for Arabic captions."""
    
    def __init__(self, api_key: str, base_url: str, model_id: str):
        """
        Initialize LLM judge.
        
        Args:
            api_key: OpenAI API key
            base_url: API base URL (e.g., OpenAI, OpenRouter)
            model_id: Model identifier to use
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_id = model_id
        self.system_prompt = config.LLM_JUDGE_CONFIG["system_prompt"]
        self.temperature = config.LLM_JUDGE_CONFIG["temperature"]
        self.max_retries = config.LLM_JUDGE_CONFIG["max_retries"]
        self.structured_output = config.LLM_JUDGE_CONFIG["structured_output"]
    
    def judge_captions(self, model_caption: str, ground_truth_caption: str, 
                      structured: Optional[bool] = None) -> Judgement:
        """
        Judge the similarity between two captions.
        
        Args:
            model_caption: Caption generated by the model
            ground_truth_caption: Ground truth caption for comparison
            structured: Whether to use structured output parsing
            
        Returns:
            Judgement object containing the similarity score, or -1 if error occurs
        """
        if structured is None:
            structured = self.structured_output
        
        for attempt in range(self.max_retries):
            try:
                if structured:
                    # Structured parsing
                    response = self.client.beta.chat.completions.parse(
                        model=self.model_id,
                        response_format=Judgement,
                        temperature=self.temperature,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {
                                "role": "user",
                                "content": f"Model Caption: {model_caption}\nGround Truth Caption: {ground_truth_caption}",
                            },
                        ],
                    )
                    return response.choices[0].message.parsed
                else:
                    # Fall back to unstructured parsing (regex int matching)
                    response = self.client.chat.completions.create(
                        model=self.model_id,
                        temperature=self.temperature,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {
                                "role": "user",
                                "content": f"Model Caption: {model_caption}\nGround Truth Caption: {ground_truth_caption}",
                            },
                        ],
                    )
                    content = response.choices[0].message.content
                    match = re.search(r"\d+", content)
                    if match:
                        score = int(match.group(0))
                        # Validate score range
                        if 0 <= score <= 10:
                            return Judgement(score=score)
                        else:
                            print(f"Score out of range (0-10): {score}")
                            if attempt == self.max_retries - 1:
                                return Judgement(score=-1)
                            continue
                    else:
                        print(f"Failed to parse score from: {content}")
                        if attempt == self.max_retries - 1:
                            return Judgement(score=-1)
                        continue
                        
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    return Judgement(score=-1)
                continue
        
        return Judgement(score=-1)
    
    def judge_single_pair(self, reference: str, candidate: str) -> int:
        """
        Judge a single caption pair.
        
        Args:
            reference: Ground truth caption
            candidate: Generated caption
            
        Returns:
            LLM judge score (1-10, or -1 for error)
        """
        judgement = self.judge_captions(candidate, reference)
        return judgement.score
    
    def evaluate_dataset(self, df: pd.DataFrame, 
                        ref_col: str = config.DEFAULT_COLUMNS["reference"],
                        cand_col: str = config.DEFAULT_COLUMNS["candidate"],
                        max_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Evaluate entire dataset using LLM judge.
        
        Args:
            df: DataFrame with reference and candidate columns
            ref_col: Name of reference column
            cand_col: Name of candidate column
            max_samples: Maximum number of samples to evaluate
            
        Returns:
            DataFrame with LLM judge scores added
        """
        # Limit samples if specified
        if max_samples:
            df_eval = df.head(max_samples).copy()
        else:
            df_eval = df.copy()
        
        print(f"Evaluating {len(df_eval)} samples with LLM judge...")
        
        # Apply LLM judge to each row
        tqdm.pandas(desc="LLM Judge Evaluation")
        
        def get_llm_score(row):
            return self.judge_single_pair(row[ref_col], row[cand_col])
        
        df_eval['llm_judge_score'] = df_eval.progress_apply(get_llm_score, axis=1)
        
        # Calculate statistics
        valid_scores = df_eval[df_eval['llm_judge_score'] != -1]['llm_judge_score']
        
        if len(valid_scores) > 0:
            print(f"\nLLM Judge Results:")
            print(f"Valid evaluations: {len(valid_scores)}/{len(df_eval)}")
            print(f"Average score: {valid_scores.mean():.2f}")
            print(f"Score distribution:")
            print(valid_scores.value_counts().sort_index())
        else:
            print("No valid LLM judge scores obtained")
        
        return df_eval


def create_llm_judge(api_key: str, base_url: str = "https://api.openai.com/v1", 
                    model_id: str = "gpt-4") -> LLMJudge:
    """
    Factory function to create LLM judge.
    
    Args:
        api_key: OpenAI API key
        base_url: API base URL
        model_id: Model identifier
        
    Returns:
        LLMJudge instance
    """
    return LLMJudge(api_key, base_url, model_id)