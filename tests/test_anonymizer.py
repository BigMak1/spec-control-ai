
from backend.app.anonymizer import anonymize
from backend.app.deanonymizer import deanonymize


class TestAnonymizePersons:
    def test_finds_full_name(self):
        text = "Проект утвердил Иванов Пётр Сергеевич в 2024 году."
        anon, pii_map = anonymize(text)
        assert "[PERSON_" in anon
        assert "Иванов" not in anon
        # Проверяем round-trip
        assert "Иванов" in deanonymize(anon, pii_map)

    def test_finds_initials_surname(self):
        text = "Ответственный: П.С. Коваль, руководитель проекта."
        anon, pii_map = anonymize(text)
        person_tokens = [k for k in pii_map if k.startswith("[PERSON_")]
        assert len(person_tokens) >= 1


class TestAnonymizeContacts:
    def test_finds_phone_plus7(self):
        text = "Контакты: +7 (351) 225-49-08 для связи."
        anon, pii_map = anonymize(text)
        assert "[TEL_" in anon
        assert "225-49-08" not in anon

    def test_finds_phone_8(self):
        text = "Телефон: 8-912-345-67-89, звонить после обеда."
        anon, pii_map = anonymize(text)
        assert "[TEL_" in anon

    def test_finds_email(self):
        text = "Писать на info@electro-serv.ru для заказа."
        anon, pii_map = anonymize(text)
        assert "[EMAIL_" in anon
        assert "info@electro-serv.ru" not in anon

    def test_finds_inn(self):
        text = "ИНН организации: 7453243220, зарегистрирована в 2015."
        anon, pii_map = anonymize(text)
        assert "[INN_" in anon
        assert "7453243220" not in anon


class TestAnonymizeAddresses:
    def test_finds_address(self):
        text = "Объект расположен по адресу г. Челябинск, ул. Энгельса, д. 44Д."
        anon, pii_map = anonymize(text)
        assert "[ADDR_" in anon


class TestAnonymizeRoundTrip:
    def test_full_round_trip(self):
        """anonymize → deanonymize должно восстановить исходный текст."""
        text = (
            "Проект разработан Козловым Дмитрием Юрьевичем. "
            "Адрес: г. Москва, ул. Ленина, д. 5. "
            "Тел: +7 (495) 123-45-67. Email: kozlov@stroy.ru. ИНН: 7712345678."
        )
        anon, pii_map = anonymize(text)
        restored = deanonymize(anon, pii_map)
        assert restored == text


class TestAnonymizeEdgeCases:
    def test_no_pii(self):
        text = "Кабель ВВГнг(А)-LS 3x2.5 мм² прокладывается в гофротрубе."
        anon, pii_map = anonymize(text)
        assert anon == text
        assert pii_map == {}

    def test_empty_text(self):
        anon, pii_map = anonymize("")
        assert anon == ""
        assert pii_map == {}

    def test_duplicate_values_same_token(self):
        """Одинаковые PII-значения должны получать один и тот же токен."""
        text = (
            "Автор: Иванов Пётр Сергеевич. "
            "Утвердил: Иванов Пётр Сергеевич."
        )
        anon, pii_map = anonymize(text)
        # Должен быть один токен для одного значения
        person_tokens = [k for k in pii_map if k.startswith("[PERSON_")]
        assert len(person_tokens) >= 1
