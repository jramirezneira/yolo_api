from flask import Flask, jsonify, request, Response
import boto3
import json
from pkg_resources import parse_version

app = Flask(__name__)
print("This line will be printed.")

s3_client = boto3.client('s3')

response = s3_client.get_object(Bucket='variosjavierramirez', Key='app.json')

object_content = response["Body"].read().decode("utf-8")

data = json.loads(object_content)

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True,  port=5001)