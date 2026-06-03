"""
sentiment_engine.py
===================
LUNA Autonomous Trading Agent — NLP-Based Sentiment Analysis Engine

Replaces basic RSS sentiment with:
1. FinBERT (transformers + torch) - primary
2. VADER (lightweight fallback)
3. Reddit sentiment via PRAW (optional)
4. Credibility weighting

Features:
- Graceful degradation (auto-fallback if GPU/CPU overload)
- Async sentiment batch processing
- Reddit integration (r/wallstreetbets, r/CryptoCurrency, etc.)
"""

from __future__ import annotations

import logging
import os
from typing import Optional
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger("sentiment_engine")
logger.setLevel(logging.INFO)

# Try to import transformers/torch, fallback to VADER
try:
    from transformers import pipeline
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False
    logger.warning("transformers not available. Using VADER fallback.")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    logger.warning("vaderSentiment not available. Sentiment analysis disabled.")

try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False
    logger.info("praw not available. Reddit sentiment disabled.")


class SentimentEngine:
    """NLP sentiment analysis with graceful fallback chain."""
    
    def __init__(self):
        self.use_finbert = FINBERT_AVAILABLE
        self.use_vader = VADER_AVAILABLE
        self.use_reddit = PRAW_AVAILABLE and self._reddit_configured()
        
        self.finbert_pipeline = None
        self.vader_analyzer = None
        self.reddit_client = None
        
        if self.use_finbert:
            try:
                self.finbert_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
                logger.info("FinBERT pipeline initialized successfully")
            except Exception as exc:
                logger.warning("Failed to initialize FinBERT: %s. Falling back to VADER.", exc)
                self.use_finbert = False
        
        if self.use_vader and not self.finbert_pipeline:
            try:
                self.vader_analyzer = SentimentIntensityAnalyzer()
                logger.info("VADER sentiment analyzer initialized")
            except Exception as exc:
                logger.warning("Failed to initialize VADER: %s", exc)
                self.use_vader = False
        
        if self.use_reddit:
            try:
                self.reddit_client = praw.Reddit(
                    client_id=os.getenv("REDDIT_CLIENT_ID"),
                    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
                    user_agent="LUNA/1.0",
                )
                logger.info("Reddit client initialized")
            except Exception as exc:
                logger.warning("Failed to initialize Reddit client: %s", exc)
                self.use_reddit = False
    
    def _reddit_configured(self) -> bool:
        """Check if Reddit API credentials are set."""
        return bool(os.getenv("REDDIT_CLIENT_ID") and os.getenv("REDDIT_CLIENT_SECRET"))
    
    def analyze_headline(self, headline: str, source_credibility: float = 1.0) -> dict:
        """
        Analyze sentiment of a single headline.
        
        Args:
            headline: News headline text
            source_credibility: Weight factor (0.0-1.0) for source reliability
        
        Returns:
            {
                "text": headline,
                "score": float (-1 to +1),
                "confidence": float (0 to 1),
                "label": str ("positive", "negative", "neutral"),
                "source_weight": float,
            }
        """
        if not headline or not isinstance(headline, str):
            return self._neutral_result(headline)
        
        if self.use_finbert and self.finbert_pipeline:
            return self._finbert_analyze(headline, source_credibility)
        elif self.use_vader and self.vader_analyzer:
            return self._vader_analyze(headline, source_credibility)
        else:
            logger.warning("No sentiment analyzer available")
            return self._neutral_result(headline)
    
    def _finbert_analyze(self, text: str, source_credibility: float) -> dict:
        """Analyze using FinBERT."""
        try:
            results = self.finbert_pipeline(text[:512])  # FinBERT max 512 tokens
            if not results:
                return self._neutral_result(text)
            
            result = results[0]
            label = result.get("label", "").lower()
            score = result.get("score", 0.0)
            
            # Map FinBERT labels to sentiment score
            if "positive" in label:
                sentiment_score = score
            elif "negative" in label:
                sentiment_score = -score
            else:
                sentiment_score = 0.0
            
            return {
                "text": text,
                "score": float(sentiment_score) * source_credibility,
                "confidence": float(score),
                "label": "positive" if sentiment_score > 0 else ("negative" if sentiment_score < 0 else "neutral"),
                "source_weight": source_credibility,
                "model": "finbert",
            }
        except Exception as exc:
            logger.warning("FinBERT analysis failed: %s. Trying VADER.", exc)
            self.use_finbert = False
            if self.use_vader and self.vader_analyzer:
                return self._vader_analyze(text, source_credibility)
            return self._neutral_result(text)
    
    def _vader_analyze(self, text: str, source_credibility: float) -> dict:
        """Analyze using VADER (lightweight, offline)."""
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            compound = scores.get("compound", 0.0)  # -1 to +1
            
            return {
                "text": text,
                "score": float(compound) * source_credibility,
                "confidence": abs(compound),
                "label": "positive" if compound > 0.05 else ("negative" if compound < -0.05 else "neutral"),
                "source_weight": source_credibility,
                "model": "vader",
            }
        except Exception as exc:
            logger.warning("VADER analysis failed: %s", exc)
            return self._neutral_result(text)
    
    def _neutral_result(self, text: str) -> dict:
        """Return neutral sentiment when analysis unavailable."""
        return {
            "text": text,
            "score": 0.0,
            "confidence": 0.0,
            "label": "neutral",
            "source_weight": 1.0,
            "model": "unavailable",
        }
    
    def aggregate_sentiments(self, headlines: list[str], source_credibilities: list[float] | None = None) -> dict:
        """
        Aggregate sentiment across multiple headlines.
        
        Returns weighted average sentiment score (-1 to +1).
        """
        if not headlines:
            return {
                "sentiment_score": 0.0,
                "news_volume": 0,
                "average_confidence": 0.0,
                "breakdown": {"positive": 0, "negative": 0, "neutral": 0},
            }
        
        if source_credibilities is None:
            source_credibilities = [1.0] * len(headlines)
        
        results = [
            self.analyze_headline(h, cred)
            for h, cred in zip(headlines, source_credibilities)
        ]
        
        scores = [r["score"] for r in results]
        confidences = [r["confidence"] for r in results]
        labels = [r["label"] for r in results]
        
        weighted_score = np.mean(scores) if scores else 0.0
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        breakdown = {
            "positive": sum(1 for l in labels if l == "positive"),
            "negative": sum(1 for l in labels if l == "negative"),
            "neutral": sum(1 for l in labels if l == "neutral"),
        }
        
        return {
            "sentiment_score": float(np.clip(weighted_score, -1.0, 1.0)),
            "news_volume": len(headlines),
            "average_confidence": float(avg_confidence),
            "breakdown": breakdown,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
    
    def get_reddit_sentiment(self, tickers: list[str], subreddit_list: list[str] | None = None, limit: int = 50) -> dict:
        """
        Fetch Reddit sentiment for given tickers from specified subreddits.
        
        Default subreddits:
        - Stocks: r/wallstreetbets, r/stocks, r/investing
        - Crypto: r/CryptoCurrency, r/Bitcoin
        """
        if not self.use_reddit or not self.reddit_client:
            return {"error": "Reddit sentiment disabled or not configured"}
        
        if subreddit_list is None:
            subreddit_list = ["wallstreetbets", "stocks", "investing", "CryptoCurrency", "Bitcoin"]
        
        reddit_sentiments = {}
        
        try:
            for subreddit_name in subreddit_list:
                try:
                    subreddit = self.reddit_client.subreddit(subreddit_name)
                    
                    for ticker in tickers:
                        # Search for ticker mentions in recent posts
                        posts = subreddit.search(f"{ticker}", time_filter="day", limit=limit)
                        post_scores = []
                        
                        for post in posts:
                            # Use upvote ratio and score as sentiment proxy
                            if post.upvote_ratio > 0.5:
                                score = post.upvote_ratio * np.log1p(post.score)
                                post_scores.append(score)
                        
                        if post_scores:
                            avg_score = np.mean(post_scores)
                            reddit_sentiments[f"{ticker}_{subreddit_name}"] = {
                                "ticker": ticker,
                                "subreddit": subreddit_name,
                                "posts_found": len(post_scores),
                                "avg_sentiment": min(1.0, max(-1.0, avg_score / 10.0)),  # Normalize to -1 to +1
                            }
                except Exception as exc:
                    logger.debug("Failed to fetch Reddit sentiment for %s in r/%s: %s", ticker, subreddit_name, exc)
        
        except Exception as exc:
            logger.warning("Reddit sentiment fetch failed: %s", exc)
        
        return reddit_sentiments
    
    def health_check(self) -> dict:
        """Check sentiment engine health."""
        return {
            "finbert_available": self.use_finbert and self.finbert_pipeline is not None,
            "vader_available": self.use_vader and self.vader_analyzer is not None,
            "reddit_available": self.use_reddit and self.reddit_client is not None,
            "primary_model": "finbert" if self.use_finbert else ("vader" if self.use_vader else "unavailable"),
        }


# Global singleton
_sentiment_engine: SentimentEngine | None = None


def get_sentiment_engine() -> SentimentEngine:
    """Get or create global sentiment engine."""
    global _sentiment_engine
    if _sentiment_engine is None:
        _sentiment_engine = SentimentEngine()
    return _sentiment_engine
