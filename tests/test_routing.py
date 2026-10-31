from backend.graph import _route, classify_intent


def test_intent_router_selects_specialist():
    assert classify_intent("民法典第577条规定是什么？") == "statute"
    assert classify_intent("找劳动争议的类似判例") == "case"
    assert classify_intent("公司欠薪该怎么维权") == "aid"
    assert classify_intent("查询劳动法并告诉我该怎么仲裁") == "mixed"


def test_model_general_intent_does_not_override_specific_match(monkeypatch):
    class Result:
        content = '{"intent":"general"}'

    class FakeModel:
        def invoke(self, messages):
            return Result()

    monkeypatch.setattr("backend.graph._llm", lambda: FakeModel())
    state = {"message": "劳动法第577条规定是什么？", "history": []}

    assert _route(state)["intent"] == "statute"
