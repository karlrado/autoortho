"""Module for handling retrieval of Apple Map access tokens"""

from re import A
import requests

class AppleTokenService:
    """Service for retrieving and renewing Apple Map access tokens"""
    def __init__(self):
        self.duckduckgo_token_url = "https://duckduckgo.com/local.js?get_mk_token=1"
        self.apple_token_url = "https://cdn.apple-mapkit.com/ma/bootstrap?apiVersion=2&mkjsVersion=5.79.95&poi=1"
        self.apple_token = None

    def process_apple_token(self, apple_token: str) -> str:
        """Process the Apple Maps access token"""
        return str(apple_token).replace("/", "%2F").replace("=", "%3D").replace("+", "%2B")

    def reset_apple_maps_token(self) -> str:
        """
        Retrieve Apple Maps access token
        This is a weird and morally questionable workaround to get the token.
        But it's functional.
        """
        try:
            dd_go_token_response = requests.get(self.duckduckgo_token_url)
            dd_go_token_response.raise_for_status()
            dd_go_token = dd_go_token_response.text

            apple_token_response = requests.get(
                url=self.apple_token_url,
                headers={
                    "Origin": "https://duckduckgo.com",
                    "Authorization": f"Bearer {dd_go_token}",
                },
            )
            apple_token_response.raise_for_status()
            apple_token_body = apple_token_response.json()
            apple_token = self.process_apple_token(apple_token_body["accessKey"])

            self.apple_token = apple_token

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to retrieve Apple Maps token: {e}")

apple_token_service = AppleTokenService()
apple_token_service.reset_apple_maps_token()

print(apple_token_service.apple_token)