"""
ai_explainer.py — AI Explanation Module for Smart Traffic Signal Controller

Provides Claude LLM-powered natural-language explanations of signal decisions,
a template-based fallback for offline use, and a chat handler for follow-up questions.
"""

import json
import os
import threading
import time
from typing import Optional


# ────────────────────────────────────────────────────────────
# Template Fallback (always available, no API needed)
# ────────────────────────────────────────────────────────────

class TemplateFallback:
    """
    Generates rule-based explanation strings when the Claude API is unavailable.
    Uses decision context to produce readable explanations.
    """

    def explain_decision(self, decision_context: dict) -> str:
        """Generate a template-based explanation for a signal decision."""
        queues = decision_context.get("queues", {})
        waits = decision_context.get("avg_wait", {})
        emergency = decision_context.get("emergency", {})
        chosen = decision_context.get("chosen_phase", "Unknown")
        scores = decision_context.get("need_scores", {})

        # Determine the winning group
        if "NORTH_SOUTH" in chosen.upper():
            winning_dirs = ["north", "south"]
            losing_dirs = ["east", "west"]
            winning_name = "North-South"
            losing_name = "East-West"
        else:
            winning_dirs = ["east", "west"]
            losing_dirs = ["north", "south"]
            winning_name = "East-West"
            losing_name = "North-South"

        # Build explanation parts
        parts = []

        # Check emergency
        has_emergency = any(emergency.get(d, False) for d in winning_dirs)
        if has_emergency:
            emg_dir = [d.capitalize() for d in winning_dirs if emergency.get(d, False)]
            parts.append(
                f"{winning_name} received priority because an emergency vehicle "
                f"was detected in the {', '.join(emg_dir)} approach."
            )
        else:
            # Queue comparison
            win_queue = sum(queues.get(d, 0) for d in winning_dirs)
            lose_queue = sum(queues.get(d, 0) for d in losing_dirs)
            win_wait = max(waits.get(d, 0) for d in winning_dirs)

            if win_queue > lose_queue:
                ratio = win_queue / max(lose_queue, 1)
                parts.append(
                    f"{winning_name} received green because its combined queue "
                    f"({win_queue} vehicles) was {ratio:.1f}x longer than {losing_name} "
                    f"({lose_queue} vehicles)."
                )
            else:
                parts.append(
                    f"{winning_name} received green due to higher average wait times "
                    f"(up to {win_wait:.1f}s)."
                )

        # Add wait time detail
        max_wait_dir = max(winning_dirs, key=lambda d: waits.get(d, 0))
        max_wait_val = waits.get(max_wait_dir, 0)
        if max_wait_val > 5:
            parts.append(
                f"The {max_wait_dir.capitalize()} approach had the longest average "
                f"wait at {max_wait_val:.1f} seconds."
            )

        return " ".join(parts)

    def answer_question(self, question: str, sim_state: dict) -> str:
        """Generate a template-based answer to a user question."""
        q_lower = question.lower()
        queues = sim_state.get("queues", {})
        waits = sim_state.get("avg_wait", {})
        phase = sim_state.get("current_phase", "Unknown")

        if "why" in q_lower and ("not" in q_lower or "instead" in q_lower):
            return (
                f"The current phase is {phase}. The controller selects the direction "
                f"group with the highest need score (queue length x wait time x emergency "
                f"multiplier). Current queues: {queues}. Current wait times: {waits}."
            )
        elif "emergency" in q_lower:
            return (
                "Emergency vehicles receive a 3x multiplier on their lane's need score, "
                "which typically forces an immediate phase switch to clear the emergency vehicle. "
                f"Current emergency status: {sim_state.get('emergency', 'none detected')}."
            )
        elif "wait" in q_lower or "time" in q_lower:
            return (
                f"Current average wait times per direction: {waits}. "
                f"The controller prioritizes directions with longer combined wait times "
                f"to minimize overall congestion."
            )
        elif "queue" in q_lower or "vehicle" in q_lower:
            return (
                f"Current queue lengths per direction: {queues}. "
                f"Longer queues receive proportionally higher need scores."
            )
        else:
            return (
                f"The controller uses a priority-based scoring system. Each tick, it "
                f"computes: need_score = queue_length x avg_wait_time x emergency_multiplier "
                f"for each direction group (N-S and E-W). The group with the higher score "
                f"gets green. Current phase: {phase}, queues: {queues}."
            )


# ────────────────────────────────────────────────────────────
# Gemini Explainer (requires Gemini API key)
# ────────────────────────────────────────────────────────────

class GeminiExplainer:
    """
    Sends decision context to Gemini API for natural-language explanations.
    Falls back to TemplateFallback on error/timeout.
    """

    SYSTEM_PROMPT = (
        "You are a traffic signal controller AI assistant. Your job is to explain "
        "signal decisions in 1-2 plain-English sentences. Reference specific numbers "
        "(queue lengths, wait times, emergency status). Be concise and informative. "
        "Do not use technical jargon. Speak as if explaining to a city traffic operator."
    )

    CHAT_SYSTEM_PROMPT = (
        "You are a traffic signal controller AI assistant. The user is watching a live "
        "simulation of a smart traffic signal. Answer their questions about the controller's "
        "behavior using the provided simulation state. Be concise (2-3 sentences max). "
        "Reference specific numbers from the state data when relevant."
    )

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-3.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = model
        self.fallback = TemplateFallback()
        self.client = None
        self._available = False
        self._chat_history = []
        self._setup_client()

    def set_api_key(self, api_key: str):
        """Update the API key at runtime and re-initialize the client."""
        self.api_key = api_key.strip()
        self._setup_client()

    def _setup_client(self):
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                self._available = True
                
                # List and print available models to diagnose
                print("[AI Explainer] Fetching available models for your API key...")
                models = list(self.client.models.list())
                print("[AI Explainer] Available models:")
                for m in models:
                    # m.name is usually like 'models/gemini-1.5-flash'
                    print(f"  - {m.name}")
            except ImportError:
                print("[AI Explainer] google-genai package not installed. Using template fallback.")
                self._available = False
            except Exception as e:
                print(f"[AI Explainer] Failed to initialize Gemini client or list models: {e}")
                self._available = False
        else:
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def explain_decision(self, decision_context: dict, callback=None) -> str:
        """
        Get a natural-language explanation for a signal decision.
        
        If callback is provided, runs asynchronously in a thread and calls
        callback(explanation_text) when done.
        
        If no callback, runs synchronously.
        """
        if callback:
            thread = threading.Thread(
                target=self._explain_threaded,
                args=(decision_context, callback),
                daemon=True,
            )
            thread.start()
            return ""  # will be delivered via callback
        else:
            return self._do_explain(decision_context)

    def _explain_threaded(self, decision_context: dict, callback):
        """Run explanation in a background thread."""
        try:
            result = self._do_explain(decision_context)
            callback(result)
        except Exception as e:
            callback(self.fallback.explain_decision(decision_context))

    def _generate_with_retry(self, prompt=None, is_chat=False, chat_history=None, context_msg=None):
        """Helper to try generating content with fallback models and transient retry for 503/429 errors."""
        from google.genai import types
        
        # Try primary model first, then fallbacks (only using models verified to be available)
        models_to_try = [
            self.model,
            "gemini-3.1-flash-lite",
            "gemini-2.0-flash-lite",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-flash-latest"
        ]
        models_to_try = list(dict.fromkeys([m for m in models_to_try if m]))
        
        last_err = None
        for m in models_to_try:
            # Try up to 3 times per model if we hit transient errors (503/429)
            for attempt in range(3):
                try:
                    if is_chat:
                        chat = self.client.chats.create(
                            model=m,
                            config=types.GenerateContentConfig(
                                system_instruction=self.CHAT_SYSTEM_PROMPT
                            )
                        )
                        if chat_history:
                            chat._history = chat_history
                        response = chat.send_message(
                            context_msg,
                            config=types.GenerateContentConfig(max_output_tokens=200)
                        )
                    else:
                        response = self.client.models.generate_content(
                            model=m,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                max_output_tokens=150,
                            )
                        )
                    text = response.text.strip()
                    if text:
                        # Successfully got response, update active model if it shifted
                        if m != self.model:
                            print(f"[AI Explainer] Shifted active model to: {m}")
                            self.model = m
                        return text
                except Exception as e:
                    last_err = e
                    err_msg = str(e).upper()
                    # If it's a transient overload (503) or rate limit (429) and we have retries left, wait and retry
                    if ("503" in err_msg or "UNAVAILABLE" in err_msg or "429" in err_msg or "EXHAUSTED" in err_msg) and attempt < 2:
                        wait_time = (attempt + 1) * 1.5
                        print(f"[AI Explainer] Model {m} hit transient error (Attempt {attempt+1}/3): {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        # Break out of the retry loop to try the next model
                        print(f"[AI Explainer] Model {m} failed permanently or out of retries: {e}.")
                        break
        
        if last_err:
            raise last_err
        return ""

    def _do_explain(self, decision_context: dict) -> str:
        """Actually call the API or fall back."""
        if not self._available or not self.client:
            return self.fallback.explain_decision(decision_context)

        try:
            prompt = (
                f"{self.SYSTEM_PROMPT}\n\n"
                "Explain this traffic signal decision in 1-2 plain-English sentences:\n\n"
                f"{json.dumps(decision_context, indent=2)}"
            )

            text = self._generate_with_retry(prompt=prompt, is_chat=False)
            return text if text else self.fallback.explain_decision(decision_context)

        except Exception as e:
            print(f"[AI Explainer] API call failed: {e}. Using fallback.")
            return self.fallback.explain_decision(decision_context)

    def answer_question(self, question: str, sim_state: dict, callback=None) -> str:
        """
        Answer a user's follow-up question about the controller's behavior.
        Uses conversation history for context continuity.
        """
        if callback:
            thread = threading.Thread(
                target=self._answer_threaded,
                args=(question, sim_state, callback),
                daemon=True,
            )
            thread.start()
            return ""
        else:
            return self._do_answer(question, sim_state)

    def _answer_threaded(self, question: str, sim_state: dict, callback):
        try:
            result = self._do_answer(question, sim_state)
            callback(result)
        except Exception:
            callback(self.fallback.answer_question(question, sim_state))

    def _do_answer(self, question: str, sim_state: dict) -> str:
        if not self._available or not self.client:
            return self.fallback.answer_question(question, sim_state)

        try:
            from google.genai import types
            # Build context message
            context_msg = (
                f"Current simulation state:\n{json.dumps(sim_state, indent=2)}\n\n"
                f"User question: {question}"
            )

            # Map our generic chat history format to Gemini format
            gemini_history = []
            for msg in self._chat_history[-10:]:
                role = "user" if msg["role"] == "user" else "model"
                gemini_history.append(
                    types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
                )
                
            answer = self._generate_with_retry(
                is_chat=True,
                chat_history=gemini_history,
                context_msg=context_msg
            )

            # Store in our generic history format
            self._chat_history.append({"role": "user", "content": context_msg})
            self._chat_history.append({"role": "assistant", "content": answer})

            # Trim history
            if len(self._chat_history) > 20:
                self._chat_history = self._chat_history[-10:]

            return answer if answer else self.fallback.answer_question(question, sim_state)

        except Exception as e:
            print(f"[AI Chat] API call failed: {e}. Using fallback.")
            return self.fallback.answer_question(question, sim_state)

    def clear_history(self):
        """Clear the chat conversation history."""
        self._chat_history.clear()

    def build_decision_context(self, decision) -> dict:
        """
        Build a JSON-serializable context dict from a PhaseDecision object.
        Used as the payload for API calls.
        """
        return {
            "tick": decision.tick,
            "queues": decision.queue_lengths,
            "avg_wait": decision.avg_wait_times,
            "emergency": {k: bool(v) for k, v in decision.emergency_flags.items()},
            "chosen_phase": decision.chosen_phase.name,
            "need_scores": decision.need_scores,
        }

    def build_sim_state(self, intersection, current_tick: int, last_decision=None) -> dict:
        """Build current simulation state dict for chat context."""
        state = {
            "current_tick": current_tick,
            "current_phase": intersection.current_phase.name,
            "queues": intersection.get_queue_snapshot(),
            "avg_wait": intersection.get_wait_snapshot(),
            "emergency": {k: bool(v) for k, v in intersection.get_emergency_snapshot().items()},
        }
        if last_decision:
            state["last_decision"] = {
                "tick": last_decision.tick,
                "chosen_phase": last_decision.chosen_phase.name,
                "need_scores": last_decision.need_scores,
                "reason": last_decision.reason,
            }
        return state


# ────────────────────────────────────────────────────────────
# Quick self-test
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test template fallback (no API key needed)
    fallback = TemplateFallback()

    test_context = {
        "tick": 42,
        "queues": {"north": 12, "south": 8, "east": 3, "west": 5},
        "avg_wait": {"north": 18.5, "south": 14.2, "east": 5.1, "west": 7.3},
        "emergency": {"north": True, "south": False, "east": False, "west": False},
        "chosen_phase": "NORTH_SOUTH_GREEN",
        "need_scores": {"north_south": 285.6, "east_west": 31.2},
    }

    print("=== Template Fallback Test ===")
    explanation = fallback.explain_decision(test_context)
    print(f"Explanation: {explanation}")
    print()

    # Test question answering
    answer = fallback.answer_question("Why not extend East-West instead?", test_context)
    print(f"Q: Why not extend East-West instead?")
    print(f"A: {answer}")
    print()

    # Test Gemini explainer initialization (will use fallback if no key)
    print("=== Gemini Explainer Test ===")
    explainer = GeminiExplainer()
    print(f"Gemini available: {explainer.is_available}")
    explanation = explainer.explain_decision(test_context)
    print(f"Explanation: {explanation}")
