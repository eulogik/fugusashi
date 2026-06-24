from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


from .router.strategies import SimilarityRouter


@dataclass
class OutcomeRecord:
    prompt: str
    routed_to: str
    confidence: float
    strategy: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0
    user_rating: Optional[int] = None
    auto_score: Optional[float] = None
    error: bool = False
    timestamp: str = ""


@dataclass
class ModelScore:
    wins: int = 0
    losses: int = 0
    total_cost: float = 0.0
    avg_latency_ms: float = 0.0
    ratings: List[int] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.5

    @property
    def avg_rating(self) -> float:
        return sum(self.ratings) / len(self.ratings) if self.ratings else 0.5


class FeedbackLoop:
    def __init__(self, data_dir: str = ".fugusashi_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.outcomes: List[OutcomeRecord] = []
        self.model_scores: Dict[str, ModelScore] = defaultdict(ModelScore)
        self._load_data()

    def _data_path(self) -> Path:
        return self.data_dir / "outcomes.jsonl"

    def _index_path(self) -> Path:
        return self.data_dir / "similarity_index.jsonl"

    def _load_data(self):
        if self._data_path().exists():
            with open(self._data_path()) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        self.outcomes.append(OutcomeRecord(**d))
                    except Exception:
                        continue
        self._rebuild_scores()

    def _save_outcome(self, outcome: OutcomeRecord):
        with open(self._data_path(), "a") as f:
            d = {k: v for k, v in outcome.__dict__.items() if v is not None}
            f.write(json.dumps(d) + "\n")

    def _rebuild_scores(self):
        self.model_scores = defaultdict(ModelScore)
        for o in self.outcomes:
            ms = self.model_scores[o.routed_to]
            if o.error:
                ms.losses += 1
            else:
                ms.wins += 1
            ms.total_cost += o.cost
            if o.user_rating is not None:
                ms.ratings.append(o.user_rating)

    def record_routing(
        self,
        prompt: str,
        routed_to: str,
        confidence: float,
        strategy: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
        latency_ms: float,
        error: bool = False,
        auto_retrain: bool = True,
        retrain_interval: int = 10,
        router: Any = None,
    ) -> OutcomeRecord:
        from datetime import datetime
        outcome = OutcomeRecord(
            prompt=prompt,
            routed_to=routed_to,
            confidence=confidence,
            strategy=strategy,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            latency_ms=latency_ms,
            error=error,
            timestamp=datetime.utcnow().isoformat(),
        )
        self.outcomes.append(outcome)
        self._save_outcome(outcome)

        ms = self.model_scores[routed_to]
        if error:
            ms.losses += 1
        else:
            ms.wins += 1
        ms.total_cost += cost

        if auto_retrain and router and len(self.outcomes) % retrain_interval == 0:
            self.build_similarity_index(router)

        return outcome

    def record_user_rating(self, outcome: OutcomeRecord, rating: int):
        outcome.user_rating = max(1, min(5, rating))
        self._save_outcome(outcome)
        ms = self.model_scores[outcome.routed_to]
        ms.ratings.append(rating)

    def build_similarity_index(self, router: SimilarityRouter):
        training = []
        for o in self.outcomes:
            if o.error:
                continue
            score = o.auto_score if o.auto_score is not None else 0.5
            if o.user_rating is not None:
                score = o.user_rating / 5.0
            training.append({
                "prompt": o.prompt,
                "model": o.routed_to,
                "score": round(score, 2),
            })
        if training:
            router.build_index(training)

    def get_retraining_data(self, min_score: float = 0.0) -> List[dict]:
        data = []
        for o in self.outcomes:
            if o.error:
                continue
            score = o.auto_score if o.auto_score is not None else 0.5
            if o.user_rating is not None:
                score = o.user_rating / 5.0
            if score >= min_score:
                data.append({
                    "prompt": o.prompt,
                    "model": o.routed_to,
                    "score": round(score, 2),
                })
        return data

    def get_model_rankings(self) -> Dict[str, Dict[str, Any]]:
        rankings = {}
        for model, ms in self.model_scores.items():
            rankings[model] = {
                "wins": ms.wins,
                "losses": ms.losses,
                "win_rate": round(ms.win_rate, 3),
                "total_cost": round(ms.total_cost, 6),
                "avg_rating": round(ms.avg_rating, 2) if ms.ratings else None,
                "rating_count": len(ms.ratings),
            }
        return rankings

    def get_stats(self) -> Dict[str, Any]:
        total = len(self.outcomes)
        errors = sum(1 for o in self.outcomes if o.error)
        total_cost = sum(o.cost for o in self.outcomes)
        total_tokens = sum(o.prompt_tokens + o.completion_tokens for o in self.outcomes)
        avg_latency = (
            sum(o.latency_ms for o in self.outcomes) / total if total > 0 else 0
        )
        rated = [o for o in self.outcomes if o.user_rating is not None]
        avg_rating = (
            sum(o.user_rating for o in rated) / len(rated) if rated else 0
        )

        return {
            "total_outcomes": total,
            "errors": errors,
            "error_rate": round(errors / total, 4) if total > 0 else 0,
            "total_cost": round(total_cost, 6),
            "total_tokens": total_tokens,
            "avg_latency_ms": round(avg_latency, 2),
            "avg_user_rating": round(avg_rating, 2),
            "rated_count": len(rated),
            "model_rankings": self.get_model_rankings(),
        }

    def save_state(self):
        state = {
            "outcomes": [
                {k: v for k, v in o.__dict__.items() if v is not None}
                for o in self.outcomes
            ],
            "model_rankings": self.get_model_rankings(),
        }
        with open(self.data_dir / "state.json", "w") as f:
            json.dump(state, f, indent=2)

    def export_training_data(self, path: str, min_score: float = 0.6):
        data = self.get_retraining_data(min_score)
        with open(path, "w") as f:
            for d in data:
                f.write(json.dumps(d) + "\n")
        return len(data)
