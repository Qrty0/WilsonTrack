"""Assistant IA (Claude) — sans réseau : l'appel à l'API est remplacé par un faux."""
from __future__ import annotations

import base64
import copy
import json
from types import SimpleNamespace

import anthropic
import numpy as np
import pytest

from wilsontrack import ai_assistant
from wilsontrack.ai_assistant import AssistantError, ClaudeAssistant, FrameVerdict

try:  # le SDK Anthropic installé s'appuie sur httpx2 (ou httpx pour les versions plus anciennes)
    import httpx2 as _httpx
except ImportError:  # pragma: no cover
    import httpx as _httpx

API_URL = "https://api.anthropic.com/v1/messages"


def _response(text: str | None, stop_reason: str = "end_turn"):
    blocks = [SimpleNamespace(type="thinking", thinking="…")]
    if text is not None:
        blocks.append(SimpleNamespace(type="text", text=text))
    return SimpleNamespace(stop_reason=stop_reason, content=blocks)


class FakeCreate:
    """Remplace client.beta.messages.create ; enregistre une copie des arguments de chaque appel."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        r = self.replies.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r


@pytest.fixture
def assistant(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    a = ClaudeAssistant(api_key="sk-ant-test")
    return a


def _install(monkeypatch, a: ClaudeAssistant, replies) -> FakeCreate:
    fake = FakeCreate(replies)
    monkeypatch.setattr(a.client.beta.messages, "create", fake)
    return fake


def _request():
    return _httpx.Request("POST", API_URL)


def _status_error(cls, code: int, message: str):
    resp = _httpx.Response(code, request=_request(), headers={"request-id": "req_test"})
    return cls(message, response=resp, body=None)


VERDICT = {
    "traces": [{"id": 1, "classe": "alpha", "confiance": 0.9, "justification": "Trace épaisse et courte."},
               {"id": 2, "classe": "artefact", "confiance": 0.6, "justification": "Reflet."}],
    "traces_non_detectees": "Une trace fine en haut à gauche.",
    "qualite_image": "Bonne",
    "conseils_montage": ["Baisser l'éclairage latéral."],
    "resume": "Deux traces dont une alpha.",
}

TRACKS = [{"id": 1, "classe": "alpha", "longueur_mm": 36.1, "epaisseur_mm": 1.4, "rectitude": 0.99,
           "densite_rel": 4.2, "duree_visible_s": 1.2},
          {"id": 2, "classe": "beta", "longueur_mm": 12.0, "epaisseur_mm": 0.5, "rectitude": 0.8,
           "densite_rel": 1.0, "duree_visible_s": 0.4}]


# --------------------------------------------------------------------------- #
def test_client_configuration(assistant):
    assert assistant.client.api_key == "sk-ant-test"
    assert assistant.model == ai_assistant.MODEL
    assert assistant.history == []


def test_analyze_frame_parses_json(monkeypatch, assistant):
    fake = _install(monkeypatch, assistant, [_response(json.dumps(VERDICT, ensure_ascii=False))])
    img = np.zeros((1080, 1920, 3), np.uint8)
    img[500:520, 900:1100] = 255
    v = assistant.analyze_frame(img, TRACKS, context="Fond, 5 min")
    assert isinstance(v, FrameVerdict)
    assert [t["classe"] for t in v.traces] == ["alpha", "artefact"]
    assert v.missed == VERDICT["traces_non_detectees"]
    assert v.quality == "Bonne"
    assert v.advice == VERDICT["conseils_montage"]
    assert v.summary == VERDICT["resume"]
    assert v.raw == VERDICT

    (call,) = fake.calls
    assert call["model"] == ai_assistant.MODEL
    assert call["system"] == ai_assistant.SYSTEM
    assert call["output_config"]["format"] == {"type": "json_schema", "schema": ai_assistant.FRAME_SCHEMA}
    assert call["output_config"]["effort"] == "medium"
    (msg,) = call["messages"]
    assert msg["role"] == "user"
    image, text = msg["content"]
    assert image["type"] == "image" and image["source"]["media_type"] == "image/jpeg"
    jpg = base64.b64decode(image["source"]["data"])
    assert jpg[:2] == b"\xff\xd8"                        # JPEG valide
    import cv2
    dec = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert max(dec.shape[:2]) <= 1568                     # image réduite pour l'API
    assert "#1: classe locale=alpha" in text["text"] and "#2" in text["text"]
    assert "Contexte : Fond, 5 min" in text["text"]


def test_analyze_frame_without_tracks(monkeypatch, assistant):
    fake = _install(monkeypatch, assistant, [_response(json.dumps(VERDICT))])
    assistant.analyze_frame(np.zeros((100, 150, 3), np.uint8), [])
    text = fake.calls[0]["messages"][0]["content"][1]["text"]
    assert "aucune trace détectée" in text


def test_analyze_frame_invalid_json(monkeypatch, assistant):
    _install(monkeypatch, assistant, [_response("ceci n'est pas du JSON")])
    with pytest.raises(AssistantError, match="illisible"):
        assistant.analyze_frame(np.zeros((50, 50, 3), np.uint8), TRACKS)


def test_chat_keeps_history(monkeypatch, assistant):
    fake = _install(monkeypatch, assistant, [_response("Réponse 1"), _response("Réponse 2")])
    assert assistant.chat("Pourquoi autant de muons ?", context="46 traces en 90 s") == "Réponse 1"
    assert assistant.chat("Et les alphas ?", context="ignoré") == "Réponse 2"
    h = assistant.history
    assert [m["role"] for m in h] == ["user", "assistant", "user", "assistant"]
    assert h[0]["content"].startswith("Contexte (résultats actuels) :\n46 traces en 90 s")
    assert h[1]["content"] == "Réponse 1"
    assert h[2]["content"] == "Et les alphas ?"          # le contexte n'est joint qu'au 1er message
    assert len(fake.calls[0]["messages"]) == 1
    assert len(fake.calls[1]["messages"]) == 3
    assert "output_config" in fake.calls[1] and "format" not in fake.calls[1]["output_config"]
    assistant.reset()
    assert assistant.history == []


def test_chat_error_does_not_pollute_history(monkeypatch, assistant):
    _install(monkeypatch, assistant, [_response("OK"), _response(None, stop_reason="refusal")])
    assistant.chat("Question 1")
    with pytest.raises(AssistantError):
        assistant.chat("Question 2")
    assert [m["content"] for m in assistant.history] == ["Question 1", "OK"]


def test_refusal_raises(monkeypatch, assistant):
    _install(monkeypatch, assistant, [_response("", stop_reason="refusal")])
    with pytest.raises(AssistantError, match="décliné"):
        assistant.interpret("Rapport")


def test_empty_answer_raises(monkeypatch, assistant):
    _install(monkeypatch, assistant, [_response(None)])
    with pytest.raises(AssistantError, match="vide"):
        assistant.test_connection()


def test_interpret_sends_report(monkeypatch, assistant):
    fake = _install(monkeypatch, assistant, [_response("## Interprétation\nTexte.")])
    out = assistant.interpret("Bêta : 25 /min contre 9 /min", question="Est-ce significatif ?")
    assert out.startswith("## Interprétation")
    content = fake.calls[0]["messages"][0]["content"]
    assert "Bêta : 25 /min contre 9 /min" in content and "Est-ce significatif ?" in content


def test_connection_error_mapped_to_french_message(monkeypatch, assistant):
    err = anthropic.APIConnectionError(request=_request())
    _install(monkeypatch, assistant, [err])
    with pytest.raises(AssistantError) as ei:
        assistant.test_connection()
    msg = str(ei.value)
    assert "connexion" in msg.lower() and "hors ligne" in msg
    assert ei.value.__cause__ is err


def test_timeout_error_mapped(monkeypatch, assistant):
    _install(monkeypatch, assistant, [anthropic.APITimeoutError(request=_request())])
    with pytest.raises(AssistantError, match="connexion"):
        assistant.chat("Bonjour")
    assert assistant.history == []


@pytest.mark.parametrize("cls,code,expected", [
    (anthropic.AuthenticationError, 401, "Clé API refusée"),
    (anthropic.PermissionDeniedError, 403, "n'a pas accès"),
    (anthropic.NotFoundError, 404, "Modèle introuvable"),
    (anthropic.RateLimitError, 429, "Limite de requêtes"),
    (anthropic.BadRequestError, 400, "Requête refusée par l'API : message API"),
    (anthropic.InternalServerError, 500, "Erreur du service Anthropic (500)"),
])
def test_status_errors_mapped(monkeypatch, assistant, cls, code, expected):
    _install(monkeypatch, assistant, [_status_error(cls, code, "message API")])
    with pytest.raises(AssistantError, match=expected.replace("(", r"\(").replace(")", r"\)")):
        assistant.interpret("Rapport")


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    # Sans clé : soit le constructeur échoue avec un message français, soit l'appel échoue plus tard
    try:
        a = ClaudeAssistant(api_key="")
    except AssistantError as e:
        assert "clé API" in str(e)
        return
    assert a.client is not None


# =========================================================================== #
#  Opus 5.5 / Sonnet 5.5 / Haiku 5.5 : routage, effort, second avis par trace  #
# =========================================================================== #
class FakeClient:
    """Faux client Anthropic : client.messages.create et client.beta.messages.create."""

    def __init__(self, replies):
        self.plain = FakeCreate(replies)
        self.beta_create = FakeCreate([])
        self.beta_create.replies = self.plain.replies       # file de réponses partagée
        self.messages = SimpleNamespace(create=self.plain)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.beta_create))


def _fake_client(monkeypatch, a: ClaudeAssistant, replies) -> FakeClient:
    fc = FakeClient(replies)
    monkeypatch.setattr(a, "client", fc)
    return fc


OPINION = {
    "indices_visuels": ["Trace fine et rectiligne", "Gouttelettes peu denses et régulières"],
    "classe": "muon",
    "confiance": 0.82,
    "pourquoi": ["Densité d'ionisation ×1,0 : minimum d'ionisation (Bethe–Bloch).",
                 "Rectitude 0,995 : diffusion multiple négligeable, particule très énergétique."],
    "pourquoi_pas": {"alpha": "Trop fine et trop pâle pour une alpha.",
                     "beta": "Aucun coude ni zigzag sur 48 mm.",
                     "muon": "Un électron énergétique reste possible.",
                     "proton": "Ionisation trop faible pour un proton lent."},
    "accord_avec_logiciel": True,
    "limites": ["Projection 2D", "Échelle non calibrée"],
    "resume": "Trace longue, fine et droite : muon cosmique très probable.",
}

MEAS = {"classe_locale": "muon", "confiance_locale": 0.9, "scores_locaux": {"muon": 0.9, "beta": 0.1},
        "longueur_mm": 48.25, "epaisseur_mm": 0.42, "epaisseur_rel": 1.05, "densite_rel": 0.98,
        "rectitude": 0.995, "fleche_rel": 0.004, "tortuosite_rad_mm": 0.012, "duree_visible_s": 1.75,
        "touche_bord": True, "calibre": False, "temperature_c": -28.5, "_path": [1, 2, 3]}


def _thumb(h=40, w=120):
    img = np.zeros((h, w, 3), np.uint8)
    img[h // 2 - 1:h // 2 + 1, 10:w - 10] = 230
    return img


def test_model_and_effort_choices():
    ids = [m for m, _ in ai_assistant.MODEL_CHOICES]
    assert ids == ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
    assert ids[0] == ai_assistant.MODEL
    assert [e for e, _ in ai_assistant.EFFORT_CHOICES] == ["low", "medium", "high", "xhigh"]
    for mid, label in ai_assistant.MODEL_CHOICES:
        assert ai_assistant.resolve_model(label) == mid
        assert ai_assistant.resolve_model(mid) == mid
    assert ai_assistant.resolve_model("Claude Opus 5.5") == "claude-opus-5-5"
    assert ai_assistant.resolve_model("") == ai_assistant.MODEL
    assert ai_assistant.resolve_model("claude-modele-perso") == "claude-modele-perso"
    for val, label in ai_assistant.EFFORT_CHOICES:
        assert ai_assistant.resolve_effort(label) == val
        assert ai_assistant.resolve_effort(val) == val
    assert ai_assistant.resolve_effort("Élevé") == "high"
    assert ai_assistant.resolve_effort("n'importe quoi") == "medium"
    assert ai_assistant.model_label("claude-opus-5-5") == "Claude Opus 5.5"


def test_constructor_accepts_labels(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    a = ClaudeAssistant("sk-ant-test", model="Claude Haiku 5.5 — économique", effort="Très élevé — le plus "
                                                                                   "rigoureux (plus lent)")
    assert a.model == "claude-haiku-5-5" and a.effort == "xhigh"


@pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-sonnet-5-5"])
def test_routing_fallback_models_use_beta(monkeypatch, assistant, model):
    assistant.model = model
    fc = _fake_client(monkeypatch, assistant, [_response("Connexion établie.")])
    assert assistant.test_connection() == "Connexion établie."
    assert fc.plain.calls == []
    (call,) = fc.beta_create.calls
    assert call["model"] == model
    assert call["betas"] == [ai_assistant.FALLBACK_BETA] == ["server-side-fallback-2026-07-01"]
    assert call["fallbacks"] == "default"
    assert call["output_config"]["effort"] == "low"          # test de connexion : effort minimal
    assert "thinking" not in call                              # réflexion adaptative implicite


@pytest.mark.parametrize("model", ["claude-haiku-5-5", "claude-modele-inconnu"])
def test_routing_plain_create_without_fallback(monkeypatch, assistant, model):
    assistant.model = model
    assistant.effort = "high"
    fc = _fake_client(monkeypatch, assistant, [_response("Bonjour")])
    assert assistant.interpret("Rapport") == "Bonjour"
    assert fc.beta_create.calls == []
    (call,) = fc.plain.calls
    assert call["model"] == model
    assert "betas" not in call and "fallbacks" not in call and "thinking" not in call
    assert call["output_config"] == {"effort": "high"}


def test_effort_adapted_to_older_models(monkeypatch, assistant):
    fc = _fake_client(monkeypatch, assistant, [_response("a"), _response("b"), _response("c")])
    assistant.model, assistant.effort = "claude-haiku-4-5", "high"     # pas de paramètre effort
    assistant.interpret("R")
    assert "output_config" not in fc.plain.calls[-1]
    assistant.model, assistant.effort = "claude-opus-4-6", "xhigh"     # xhigh inconnu → high
    assistant.interpret("R")
    assert fc.plain.calls[-1]["output_config"] == {"effort": "high"}
    assistant.model, assistant.effort = "claude-opus-5-5", "Moyen — équilibré"   # libellé accepté
    assistant.interpret("R")
    assert fc.beta_create.calls[-1]["output_config"] == {"effort": "medium"}


def test_explain_track_request(monkeypatch, assistant):
    assistant.effort = "high"
    fc = _fake_client(monkeypatch, assistant, [_response(json.dumps(OPINION, ensure_ascii=False))])
    assistant.explain_track(_thumb(), MEAS, "Trace fine et droite qui traverse le champ : muon.",
                            context="Bruit de fond, 10 min")
    (call,) = fc.beta_create.calls
    oc = call["output_config"]
    assert oc["effort"] == "high"
    assert oc["format"] == {"type": "json_schema", "schema": ai_assistant.TRACK_SCHEMA}
    schema = oc["format"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"classe", "confiance", "accord_avec_logiciel", "pourquoi", "pourquoi_pas",
                                       "indices_visuels", "limites", "resume"}
    assert schema["properties"]["classe"]["enum"] == ["alpha", "beta", "muon", "proton", "incertain", "artefact"]
    pp = schema["properties"]["pourquoi_pas"]
    assert pp["additionalProperties"] is False
    assert pp["required"] == ["alpha", "beta", "muon", "proton"] == list(pp["properties"])
    assert schema["properties"]["accord_avec_logiciel"]["type"] == "boolean"
    assert "minimum" not in json.dumps(schema) and "maximum" not in json.dumps(schema)

    (msg,) = call["messages"]
    image, text = msg["content"]                       # l'image AVANT le texte
    assert image["type"] == "image" and image["source"]["type"] == "base64"
    assert image["source"]["media_type"] == "image/jpeg"
    jpg = base64.b64decode(image["source"]["data"])
    assert jpg[:2] == b"\xff\xd8"
    import cv2
    dec = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert max(dec.shape[:2]) >= 700                   # vignette 120 px agrandie (≈ 768 px)
    assert max(dec.shape[:2]) <= 1568

    t = text["text"]
    assert "48,2 mm" in t or "48,3 mm" in t           # virgule décimale
    assert "0,42 mm" in t and "×0,98" in t and "×1,05" in t
    assert "0,995" in t and "0,004" in t and "0,012 rad/mm" in t and "1,75 s" in t
    assert "Touche le bord du champ : oui" in t
    assert "NON calibrée" in t
    assert "-28,5 °C" in t
    assert "INDÉPENDANT" in t and "Bethe–Bloch" in t and "diffusion multiple" in t
    assert "muon très incliné" in t and "électron énergétique" in t      # ambiguïté classique
    assert "projection 2D" in t
    assert "Trace fine et droite qui traverse le champ : muon." in t
    assert "Bruit de fond, 10 min" in t
    assert "agrandie" in t
    assert t.index("Méthode demandée") < t.index("Décision du logiciel")
    assert "_path" not in t                            # clés internes ignorées


def test_explain_track_parses_opinion(monkeypatch, assistant):
    _fake_client(monkeypatch, assistant, [_response(json.dumps(OPINION, ensure_ascii=False))])
    op = assistant.explain_track(_thumb(), MEAS, "muon")
    assert isinstance(op, ai_assistant.TrackOpinion)
    assert op.cls == "muon" and op.confidence == pytest.approx(0.82) and op.agrees is True
    assert op.why == OPINION["pourquoi"]
    assert op.why_not == OPINION["pourquoi_pas"]
    assert op.visual_clues == OPINION["indices_visuels"]
    assert op.limits == OPINION["limites"]
    assert op.summary == OPINION["resume"]
    assert op.raw == OPINION
    assert op.local_cls == "muon"
    assert op.model == "claude-opus-5-5"


def test_explain_track_sanitizes_values(monkeypatch, assistant):
    bad = dict(OPINION, classe="Bêta", confiance=85, pourquoi="Une seule raison")
    bad["pourquoi_pas"] = {"alpha": "Trop fine"}
    resp = _response("```json\n" + json.dumps(bad, ensure_ascii=False) + "\n```")
    resp.model = "claude-opus-5"                       # réponse servie par le modèle de repli
    _fake_client(monkeypatch, assistant, [resp])
    op = assistant.explain_track(_thumb(), {}, "")
    assert op.cls == "beta"
    assert op.confidence == pytest.approx(0.85)
    assert op.why == ["Une seule raison"]
    assert op.why_not == {"alpha": "Trop fine", "beta": "", "muon": "", "proton": ""}
    assert op.model == "claude-opus-5" == assistant.last_model


def test_explain_track_without_thumbnail(monkeypatch, assistant):
    fc = _fake_client(monkeypatch, assistant, [_response(json.dumps(OPINION))])
    assistant.explain_track(None, MEAS, "Explication locale")
    (block,) = fc.beta_create.calls[0]["messages"][0]["content"]
    assert block["type"] == "text"
    assert "aucune image" in block["text"]


@pytest.mark.parametrize("img", [
    np.full((30, 80), 200, np.uint8),                      # niveaux de gris
    np.random.default_rng(0).random((25, 60, 3)).astype(np.float32),   # flottant 0..1
    np.zeros((20, 50, 4), np.uint8),                       # BGRA
])
def test_explain_track_thumbnail_formats(monkeypatch, assistant, img):
    fc = _fake_client(monkeypatch, assistant, [_response(json.dumps(OPINION))])
    assistant.explain_track(img, MEAS, "")
    image = fc.beta_create.calls[0]["messages"][0]["content"][0]
    import cv2
    dec = cv2.imdecode(np.frombuffer(base64.b64decode(image["source"]["data"]), np.uint8), cv2.IMREAD_COLOR)
    assert max(dec.shape[:2]) >= 700 and min(dec.shape[:2]) >= 150


def test_explain_track_refusal(monkeypatch, assistant):
    resp = _response(None, stop_reason="refusal")
    resp.stop_details = SimpleNamespace(type="refusal", category="bio", explanation=None)
    _fake_client(monkeypatch, assistant, [resp])
    with pytest.raises(AssistantError, match="décliné") as ei:
        assistant.explain_track(_thumb(), MEAS, "")
    assert "bio" in str(ei.value) and "Reformulez" in str(ei.value)


def test_explain_track_max_tokens(monkeypatch, assistant):
    _fake_client(monkeypatch, assistant, [_response('{"classe": "mu', stop_reason="max_tokens")])
    with pytest.raises(AssistantError, match="coupée") as ei:
        assistant.explain_track(_thumb(), MEAS, "")
    assert "Réessayez" in str(ei.value)


def test_max_tokens_in_chat_keeps_history_clean(monkeypatch, assistant):
    _fake_client(monkeypatch, assistant, [_response("Début de répo", stop_reason="max_tokens")])
    with pytest.raises(AssistantError, match="coupée"):
        assistant.chat("Question longue")
    assert assistant.history == []


def test_explain_track_invalid_json(monkeypatch, assistant):
    _fake_client(monkeypatch, assistant, [_response("pas du JSON du tout")])
    with pytest.raises(AssistantError, match="illisible"):
        assistant.explain_track(_thumb(), MEAS, "")


def test_explain_track_with_haiku_routes_plain(monkeypatch, assistant):
    assistant.model = "claude-haiku-5-5"
    fc = _fake_client(monkeypatch, assistant, [_response(json.dumps(OPINION))])
    op = assistant.explain_track(_thumb(), MEAS, "")
    assert op.cls == "muon"
    assert fc.beta_create.calls == []
    call = fc.plain.calls[0]
    assert "betas" not in call and "fallbacks" not in call
    assert call["output_config"]["format"]["schema"] == ai_assistant.TRACK_SCHEMA


NAMES = {"alpha": "Alpha (α)", "beta": "Bêta / électron (β⁻)", "muon": "Muon cosmique (μ)",
         "proton": "Proton / particule lourde", "incertain": "Incertaine"}


def test_opinion_markdown():
    op = ai_assistant._opinion_from_dict(dict(OPINION, accord_avec_logiciel=False), model="claude-opus-5-5",
                                         local_cls="beta")
    md = ai_assistant.opinion_markdown(op, NAMES)
    assert md.strip()
    assert "Muon cosmique (μ)" in md                   # classe retenue
    for other in ("Alpha (α)", "Bêta / électron (β⁻)", "Proton / particule lourde"):
        assert other in md                             # section « pourquoi pas »
    assert "82 %" in md
    assert "Classe du logiciel :** Bêta / électron (β⁻)" in md and "en désaccord" in md
    assert "Pourquoi c'est : Muon cosmique (μ)" in md
    assert "Pourquoi pas les autres classes" in md
    assert "Indices visuels" in md and "Limites et incertitudes" in md
    assert OPINION["resume"] in md and "Claude Opus 5.5" in md
    # la classe retenue n'apparaît pas dans « pourquoi pas »
    assert "**Muon cosmique (μ)** :" not in md


def test_opinion_markdown_default_names_and_artefact():
    d = dict(OPINION, classe="artefact", accord_avec_logiciel=True)
    op = ai_assistant._opinion_from_dict(d)
    md = ai_assistant.opinion_markdown(op)
    assert "Artefact" in md and "pas une particule" in md
    assert "Alpha (α)" in md and "Muon cosmique (μ)" in md
    assert "Accord avec le logiciel :** oui" in md


def test_track_measurements_from_event():
    ev = SimpleNamespace(
        cls="alpha", confidence=0.93, scores={"alpha": 0.93, "proton": 0.05}, manual_cls="",
        features={"length_mm": 41.2, "chord_mm": 40.9, "width_mm": 1.31, "width_rel": 3.4, "density_rel": 5.2,
                  "straightness": 0.992, "sagitta_ratio": 0.01, "line_rms_mm": 0.12, "tortuosity": 0.02,
                  "radius_mm": float("inf"), "peak_snr": 18.0, "orientation_deg": 33.0, "branches": 0,
                  "energy_mev": 5.4, "momentum_mev": 0.0},
        duration_s=2.4, n_seen=31, touches_border=False, extra={"calibrated": True}, px_per_mm_full=7.5)
    m = ai_assistant.track_measurements(ev, temperature_c=-30.0)
    assert m["classe_locale"] == "alpha" and m["longueur_mm"] == 41.2 and m["densite_rel"] == 5.2
    assert m["fleche_mm"] == pytest.approx(0.412)
    assert m["calibre"] is True and m["px_par_mm"] == 7.5 and m["temperature_c"] == -30.0
    assert "impulsion_MeV_c" not in m                  # 0 = non mesurée
    lines, local, _ = ai_assistant._measurement_lines(m)
    txt = "\n".join(lines)
    assert "Longueur (profil à mi-hauteur) : 41,2 mm" in txt
    assert "Rayon de courbure : infini" in txt
    assert "calibrée (règle filmée" in txt
    assert "-30,0 °C" in txt
    assert local["cls"] == "alpha" and local["conf"] == 0.93


def test_system_prompt_second_opinion():
    s = ai_assistant.SYSTEM
    for word in ("SECOND AVIS", "Bethe–Bloch", "diffusion multiple", "portée–énergie", "minimum d'ionisation",
                 "projection 2D"):
        assert word in s
