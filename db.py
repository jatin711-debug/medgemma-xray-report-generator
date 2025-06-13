from http.client import HTTPException
import mysql.connector
from dotenv import load_dotenv
import os
load_dotenv(dotenv_path=".env")

def store_analysis_result(image_uuid: str, reportPath: str) -> None:
    """
    Store the analysis result in the MySQL database.

    Args:
        image_uuid: Unique identifier for the image.
        reportPath: Path to the generated PDF report.
    """
    try:
        # Connect to the MySQL database
        connection = mysql.connector.connect(
            host="your_mysql_host",
            user="your_mysql_user",
            password="your_mysql_password",
            database="your_database_name",
            port=3306  # Default MySQL port, change if necessary
        )
        cursor = connection.cursor()

        # Insert the data into the table
        query = "INSERT INTO image_reports (image_uuid, report_path) VALUES (%s, %s)"
        cursor.execute(query, (image_uuid, reportPath))
        # Commit the transaction
        connection.commit()
        cursor.close()
        connection.close()
    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail="Failed to store analysis result in database")
    
def initialize_database() -> None:
    """
    Initialize the database by creating the necessary table if it does not exist.
    """
    try:
        # Connect to the MySQL database
        connection = mysql.connector.connect(
            host=os.getenv('MYSQL_HOST'),
            user=os.getenv('MYSQL_USER'),
            password=os.getenv('MYSQL_PASSWORD'),
            database=os.getenv('MYSQL_DATABASE'),
            port=int(os.getenv('MYSQL_PORT', 3306))  # Default MySQL port, change if necessary
        )
        cursor = connection.cursor()

        # Create the table if it does not exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_reports (
                id INT AUTO_INCREMENT PRIMARY KEY,
                image_uuid VARCHAR(255) NOT NULL,
                report_path VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.commit()
        cursor.close()
        connection.close()
    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail="Failed to initialize database")