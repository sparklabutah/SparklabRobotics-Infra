import os

# Domain 92: nothing else of ours lives there. Set before rclpy first starts.
os.environ.update(ROS_DOMAIN_ID="92", ROS_LOCALHOST_ONLY="1")
