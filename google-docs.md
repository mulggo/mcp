# MCP - Google Docs

[taylorwilsdon - google_workspace_mcp](https://github.com/taylorwilsdon/google_workspace_mcp)은 Google Docs뿐 아니라 Google Calendar, Google Drive, Gmail, Google Sheets, Google Slides등을 지원하고 있습니다. 이때의 json 설정 정보는 아래와 같습니다.

```java
{
  "mcpServers": {
    "google_workspace": {
      "command": "uvx",
      "args": ["workspace-mcp"],
      "env": {
        "GOOGLE_OAUTH_CLIENT_ID": "your-client-id.apps.googleusercontent.com",
        "GOOGLE_OAUTH_CLIENT_SECRET": "your-client-secret",
        "OAUTHLIB_INSECURE_TRANSPORT": "1"
      }
    }
  }
}
```

Google OAUTH를 위해 Client ID와 Client Secret가 필요합니다. 아래와 같이 생성합니다.

1. [액세스 사용자 인증 정보 만들기](https://developers.google.com/workspace/guides/create-credentials?hl=ko)에서 API key를 생성합니다.
2. 앱에서 사용자 데이터에 액세스하려면 OAuth 2.0 클라이언트 ID를 하나 이상 만들어야 합니다. 클라이언트 만들기를 선택하여 Client ID와 Client Secret을 만들고 저장합니다.

