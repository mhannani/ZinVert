from requests.structures import CaseInsensitiveDict


def fetch(session, url, data):
    try:
        headers = CaseInsensitiveDict()
        headers["Content-Type"] = "application/json"
        result = session.post(url, data, headers=headers)
        return result.json()
    except ValueError:
        return {'Error': 'Something went wrong !'}
