# deploy/aws_agent.py
import boto3
import subprocess
import os

class AWSDeploymentAgent:
    def __init__(self, region="us-east-1"):
        self.region = region
        self.ecr = boto3.client("ecr", region_name=region)
        self.ecs = boto3.client("ecs", region_name=region)
        self.elasticache = boto3.client("elasticache", region_name=region)
    
    def run(self):
        steps = [
            self.create_ecr_repo,
            self.build_and_push_docker,
            self.create_redis_cluster,
            self.create_ecs_cluster,
            self.deploy_fargate_service,
        ]
        for step in steps:
            print(f"Running: {step.__name__}...")
            step()
    
    def create_ecr_repo(self):
        self.ecr.create_repository(repositoryName="insurance-claims-ai")
    
    def build_and_push_docker(self):
        account_id = boto3.client("sts").get_caller_identity()["Account"]
        repo_uri = f"{account_id}.dkr.ecr.{self.region}.amazonaws.com/insurance-claims-ai"
        
        subprocess.run(f"aws ecr get-login-password | docker login --username AWS --password-stdin {repo_uri}", shell=True)
        subprocess.run(f"docker build -t {repo_uri}:latest .", shell=True)
        subprocess.run(f"docker push {repo_uri}:latest", shell=True)
    
    def create_ecs_cluster(self):
        self.ecs.create_cluster(clusterName="insurance-ai-cluster")
    
    # ... etc

if __name__ == "__main__":
    agent = AWSDeploymentAgent()
    agent.run()