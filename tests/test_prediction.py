from django.test import TestCase


class TestCalls(TestCase):
    def test_call_view_loads(self):
        response = self.client.get('/api/v1/predictions/?data=PG is awesome company')
        self.assertEqual(response.status_code, 200)
