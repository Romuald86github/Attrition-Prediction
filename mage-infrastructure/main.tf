provider "aws" {
  region = "us-east-1"
}

resource "aws_db_instance" "mage_db" {
  engine         = "postgres"
  engine_version = "12.7"
  instance_class = "db.t3.micro"
  db_name        = "mage_db"
  username       = "mage_user"
  password       = "mage_password"
}

resource "aws_s3_bucket" "mage_artifacts" {
  bucket = "my-mage-artifacts"
  acl    = "private"
}

resource "aws_ecs_cluster" "mage_cluster" {
  name = "mage-cluster"
}

resource "aws_ecs_task_definition" "mage_server" {
  family                   = "mage-server"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 256
  memory                   = 512
  container_definitions    = <<DEFINITION
[
  {
    "name": "mage-server",
    "image": "mageai/mage-server:latest",
    "essential": true,
    "portMappings": [
      {
        "containerPort": 6789,
        "hostPort": 6789
      }
    ],
    "environment": [
      {
        "name": "MAGE_DB_HOST",
        "value": "${aws_db_instance.mage_db.endpoint}"
      },
      {
        "name": "MAGE_DB_NAME",
        "value": "mage_db"
      },
      {
        "name": "MAGE_DB_USER",
        "value": "mage_user"
      },
      {
        "name": "MAGE_DB_PASSWORD",
        "value": "mage_password"
      },
      {
        "name": "MAGE_ARTIFACT_STORE",
        "value": "s3://${aws_s3_bucket.mage_artifacts.id}"
      }
    ]
  }
]
DEFINITION
}

resource "aws_ecs_service" "mage_server_service" {
  name            = "mage-server-service"
  cluster         = aws_ecs_cluster.mage_cluster.id
  task_definition = aws_ecs_task_definition.mage_server.arn
  desired_count   = 1
  launch_type     = "FARGATE"
}
