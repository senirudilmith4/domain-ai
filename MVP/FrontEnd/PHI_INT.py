# PHI (Probabilistic Helpful Interventions) System
# ============================================================================

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class GPAInterventionSystem:
    """
    Probabilistic Helpful Interventions system for personalized GPA improvement recommendations.
    """

    def __init__(self, model, data, target_year: int, db=None):
        self.model = model
        self.target_year = target_year
        self.feature_names = model.feature_names_in_.tolist()
        self.gpa_threshold = self._compute_threshold(data, target_year, db)
        self.intervention_domains = self._define_intervention_domains()

    # ── THRESHOLD COMPUTATION ─────────────────────────────────────────────
    def _compute_threshold(self, data, target_year: int, db) -> float:
        """Determine GPA threshold via DB → CSV → default fallback."""
        try:
            if db is not None:
                active_data = db.load_benchmarks(target_year)
                if not active_data.empty:
                    return active_data.iloc[:, 0].quantile(0.50)
            raise ValueError("No DB data")
        except Exception:
            if data is not None:
                active_data = data[data[f'Year{target_year}_GPA'] > 0]
                return active_data[f'Year{target_year}_GPA'].quantile(0.50)
            return 2.5   # hard-coded last resort

    # ── INTERVENTION DOMAIN DEFINITIONS ──────────────────────────────────
    def _define_intervention_domains(self) -> Dict:
        """Define all possible intervention domains and their rules."""
        return {
            'time_management': {
                'feature': 'hours_per_week',
                'interventions': [
                    {
                        'name': 'Increase Study Hours (Low to Medium)',
                        'condition': lambda x: x < 4.0,
                        'change': lambda x: 7.5,
                        'description': 'Increase weekly study time to 5-10 hours using gap times between classes',
                        'icon': '📚'
                    },
                    {
                        'name': 'Increase Study Hours (Medium to High)',
                        'condition': lambda x: 4.0 <= x < 7.5,
                        'change': lambda x: 12.5,
                        'description': 'Increase weekly study time to over 10 hours with structured schedule',
                        'icon': '📚'
                    }
                ]
            },
            'procrastination': {
                'feature': 'start_assignments_closer_deadline',
                'interventions': [
                    {
                        'name': 'Reduce Procrastination (Always → Sometimes)',
                        'condition': lambda x: x == 3,
                        'change': lambda x: 2,
                        'description': 'Start assignments 3-5 days before deadline instead of last minute',
                        'icon': '⏰'
                    },
                    {
                        'name': 'Reduce Procrastination (Sometimes → Rarely)',
                        'condition': lambda x: x == 2,
                        'change': lambda x: 1,
                        'description': 'Consistently start assignments early - at least 1 week before deadline',
                        'icon': '⏰'
                    }
                ]
            },
            'task_management': {
                'feature': 'struggle_with_managing',
                'interventions': [
                    {
                        'name': 'Improve Task Management (Often → Sometimes)',
                        'condition': lambda x: x == 3,
                        'change': lambda x: 2,
                        'description': 'Use planning tools (calendar, to-do lists) to reduce management struggles',
                        'icon': '📋'
                    },
                    {
                        'name': 'Improve Task Management (Sometimes → Rarely)',
                        'condition': lambda x: x == 2,
                        'change': lambda x: 1,
                        'description': 'Master time-blocking and prioritization techniques (Eisenhower Matrix)',
                        'icon': '📋'
                    }
                ]
            },
            'stress_management': {
                'feature': 'academic_stress',
                'interventions': [
                    {
                        'name': 'Reduce Stress (High → Moderate)',
                        'condition': lambda x: x >= 4,
                        'change': lambda x: 3,
                        'description': 'Implement stress reduction: regular breaks, exercise, sleep hygiene, counseling',
                        'icon': '🧘'
                    },
                    {
                        'name': 'Reduce Stress (Moderate → Low)',
                        'condition': lambda x: x == 3,
                        'change': lambda x: 2,
                        'description': 'Enhance coping strategies: mindfulness, study groups, time management',
                        'icon': '🧘'
                    }
                ]
            },
            'programming_skill': {
                'feature': 'Skill_Programming',
                'interventions': [
                    {
                        'name': 'Improve Programming Skills',
                        'condition': lambda x: x <= 3,
                        'change': lambda x: min(5, x + 1),
                        'description': 'Dedicate 5-7 hours/week to coding practice, online courses (CS50, freeCodeCamp)',
                        'icon': '💻'
                    }
                ]
            },
            'math_skill': {
                'feature': 'Skill_Math',
                'interventions': [
                    {
                        'name': 'Improve Math Skills',
                        'condition': lambda x: x <= 3,
                        'change': lambda x: min(5, x + 1),
                        'description': 'Regular practice, attend office hours, form study groups for math courses',
                        'icon': '📐'
                    }
                ]
            },
            'workload_balance': {
                'feature': 'Balancing multiple courses or projects',
                'interventions': [
                    {
                        'name': 'Improve Course/Project Balance',
                        'condition': lambda x: x == 1,
                        'change': lambda x: 0,
                        'description': 'Create a master schedule with dedicated time blocks for each course/project',
                        'icon': '⚖️'
                    }
                ]
            },
            'task_prioritization': {
                'feature': 'Not knowing how to prioritize task',
                'interventions': [
                    {
                        'name': 'Learn Task Prioritization',
                        'condition': lambda x: x == 1,
                        'change': lambda x: 0,
                        'description': 'Learn prioritization frameworks (urgent/important matrix, ABC method)',
                        'icon': '🎯'
                    }
                ]
            },
            'deadline_management': {
                'feature': 'Starting tasks too late',
                'interventions': [
                    {
                        'name': 'Start Tasks Earlier',
                        'condition': lambda x: x == 1,
                        'change': lambda x: 0,
                        'description': 'Set personal deadlines 3-5 days before actual deadlines, use backward planning',
                        'icon': '📅'
                    }
                ]
            }
        }

    # ── CORE METHODS ──────────────────────────────────────────────────────
    def predict_gpa(self, student_data: pd.DataFrame) -> float:
        """Predict GPA ensuring correct column order."""
        student_data = student_data[self.feature_names]
        return self.model.predict(student_data)[0]

    def classify_gpa_probability(self, predicted_gpa: float) -> float:
        """Convert GPA to P(High GPA) via sigmoid."""
        distance = predicted_gpa - self.gpa_threshold
        return 1 / (1 + np.exp(-2 * distance))

    def calculate_fpp(self, current_prob: float, modified_prob: float) -> float:
        """Calculate Fold Change in Posterior Probability (FPP)."""
        p_low_current   = 1 - current_prob  if current_prob  < 0.5 else current_prob
        p_high_modified = modified_prob      if modified_prob > 0.5 else 1 - modified_prob
        p_low_current   = max(p_low_current, 0.01)   # avoid division by zero
        return p_high_modified / p_low_current

    def simulate_intervention(self, student_data: Dict,
                               domain_name: str,
                               intervention_idx: int) -> Dict:
        """Simulate a single intervention and return its impact."""
        domain       = self.intervention_domains[domain_name]
        intervention = domain['interventions'][intervention_idx]
        feature      = domain['feature']
        current_val  = student_data[feature]

        if not intervention['condition'](current_val):
            return None

        # Build modified copy
        modified_data          = student_data.copy()
        modified_data[feature] = intervention['change'](current_val)

        current_gpa  = self.predict_gpa(pd.DataFrame([student_data]))
        modified_gpa = self.predict_gpa(pd.DataFrame([modified_data]))

        current_prob  = self.classify_gpa_probability(current_gpa)
        modified_prob = self.classify_gpa_probability(modified_gpa)
        fpp           = self.calculate_fpp(current_prob, modified_prob)
        gpa_change    = modified_gpa - current_gpa

        return {
            'name':            intervention['name'],
            'description':     intervention['description'],
            'icon':            intervention['icon'],
            'domain':          domain_name,
            'fpp_score':       fpp,
            'current_gpa':     current_gpa,
            'modified_gpa':    modified_gpa,
            'gpa_improvement': gpa_change,
            'is_phi':          (fpp >= 1.0) and (gpa_change > 0)
        }

    def generate_interventions(self, student_data: Dict,
                                top_k: int = 5) -> Tuple[float, List[Dict]]:
        """Run all interventions and return the top PHI recommendations."""
        current_gpa = self.predict_gpa(pd.DataFrame([student_data]))
        results     = []

        for domain_name, domain in self.intervention_domains.items():
            for idx in range(len(domain['interventions'])):
                result = self.simulate_intervention(student_data, domain_name, idx)
                if result:
                    results.append(result)

        results.sort(key=lambda x: x['fpp_score'], reverse=True)
        phi_only = [r for r in results if r['is_phi']][:top_k]

        return current_gpa, phi_only