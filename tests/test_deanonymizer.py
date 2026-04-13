from backend.app.deanonymizer import deanonymize, find_unreplaced_tokens


def test_basic_replacement():
    text = "Проект выполнен [PERSON_1] по адресу [ADDR_1]."
    pii_map = {"[PERSON_1]": "Иванов И.И.", "[ADDR_1]": "г. Москва, ул. Ленина"}
    result = deanonymize(text, pii_map)
    assert result == "Проект выполнен Иванов И.И. по адресу г. Москва, ул. Ленина."


def test_empty_pii_map():
    text = "Текст без PII-токенов."
    assert deanonymize(text, {}) == text


def test_empty_text():
    assert deanonymize("", {"[PERSON_1]": "Иванов"}) == ""


def test_multiple_same_token():
    text = "[PERSON_1] утвердил проект. [PERSON_1] подписал документ."
    pii_map = {"[PERSON_1]": "Петров А.Б."}
    result = deanonymize(text, pii_map)
    assert result == "Петров А.Б. утвердил проект. Петров А.Б. подписал документ."


def test_all_token_categories():
    text = "[PERSON_1], [ADDR_1], [TEL_1], [EMAIL_1], [INN_1]"
    pii_map = {
        "[PERSON_1]": "Сидоров",
        "[ADDR_1]": "г. Казань",
        "[TEL_1]": "+7 123 456 78 90",
        "[EMAIL_1]": "test@mail.ru",
        "[INN_1]": "1234567890",
    }
    result = deanonymize(text, pii_map)
    assert "Сидоров" in result
    assert "г. Казань" in result
    assert "+7 123 456 78 90" in result
    assert "test@mail.ru" in result
    assert "1234567890" in result


def test_no_collision_with_similar_tokens():
    """[ADDR_1] не должен заменять часть [ADDR_10]."""
    text = "Адрес 1: [ADDR_1]. Адрес 10: [ADDR_10]."
    pii_map = {"[ADDR_1]": "Казань", "[ADDR_10]": "Москва"}
    result = deanonymize(text, pii_map)
    assert result == "Адрес 1: Казань. Адрес 10: Москва."


def test_find_unreplaced_tokens_clean():
    text = "Текст без токенов, всё заменено."
    assert find_unreplaced_tokens(text) == []


def test_find_unreplaced_tokens_found():
    text = "Отчёт: [PERSON_1] на адресе [ADDR_2], тел. [TEL_3]."
    tokens = find_unreplaced_tokens(text)
    assert set(tokens) == {"[PERSON_1]", "[ADDR_2]", "[TEL_3]"}
