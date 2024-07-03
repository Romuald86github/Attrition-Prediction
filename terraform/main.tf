provider "aws" {
     region = "us-east-1"
   }

   resource "aws_s3_bucket" "model_artifacts" {
     bucket = "my-model-artifacts"
     acl    = "private"
   }

   resource "aws_cloudwatch_log_group" "eb_logs" {
     name = "my-app-logs"
   }

   resource "aws_elastic_beanstalk_application" "my_app" {
     name = "my-app"
   }

   resource "aws_elastic_beanstalk_environment" "my_app_env" {
     name                = "my-app-env"
     application         = aws_elastic_beanstalk_application.my_app.name
     solution_stack_name = "64bit Amazon Linux 2 v3.4.6 running Python 3.8"

     setting {
       namespace = "aws:autoscaling:launchconfiguration"
       name      = "EC2KeyName"
       value     = "var.ec2_key_pair_name"
     }

     setting {
       namespace = "aws:ec2:vpc"
       name      = "VPCId"
       value     = "vpc-0123456789abcdef"
     }

     setting {
       namespace = "aws:ec2:vpc"
       name      = "Subnets"
       value     = "subnet-0123456789abcdef,subnet-fedcba9876543210"
     }

     setting {
       namespace = "aws:elasticbeanstalk:environment"
       name      = "ContainerPort"
       value     = "5000"
     }

     setting {
       namespace = "aws:elasticbeanstalk:environment"
       name      = "DockerRunArgs"
       value     = "-p 5000:5000"
     }

     setting {
       namespace = "aws:elasticbeanstalk:environment"
       name      = "DockerConfigurationSource"
       value     = "DockerRunArgs"
     }

     setting {
       namespace = "aws:elasticbeanstalk:environment"
       name      = "DockerRunImage"
       value     = "YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/my-flask-app:latest"
     }
   }
