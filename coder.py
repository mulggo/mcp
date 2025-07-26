import boto3
import time

client = boto3.client("bedrock-agentcore", region_name="us-west-2",
endpoint_url="https://bedrock-agentcore.us-west-2.amazonaws.com")

sessionId = None
response = client.list_code_interpreter_sessions(
    codeInterpreterIdentifier='aws.codeinterpreter.v1',
    maxResults=5,
    status='READY'
)
items = response['items']

if items is not None:
    # get the sessionId
    for item in items:
        sessionId = item['sessionId']
        print(f"sessionId: {sessionId}")
        break
    print(f"sessionId: {sessionId}")
else:
    print("No sessions found")
    response = client.start_code_interpreter_session(
        codeInterpreterIdentifier='aws.codeinterpreter.v1',
        name="my-code-session",
        sessionTimeoutSeconds=900
    )
    print(f"response of start_code_interpreter_session: {response}")
    time.sleep(5)

if sessionId is not None:
    response = client.get_code_interpreter_session(
        codeInterpreterIdentifier='aws.codeinterpreter.v1',
        sessionId=sessionId
    )
    print(f"response of get_code_interpreter_session: {response}")

execute_response = client.invoke_code_interpreter(
    codeInterpreterIdentifier="aws.codeinterpreter.v1",
    sessionId=sessionId,
    name="executeCode",
    arguments={
        "language": "python",
        "code": "print('Hello, World!')"
    }
)

# Extract and print the text output from the stream
for event in execute_response['stream']:
    if 'result' in event:
        result = event['result']
        if 'content' in result:
            for content_item in result['content']:
                if content_item['type'] == 'text':
                    print(content_item['text'])

# Don't forget to stop the session when you're done
response =client.stop_code_interpreter_session(
    codeInterpreterIdentifier="aws.codeinterpreter.v1",
    sessionId=sessionId
)
print(f"response of stop_code_interpreter_session: {response}")