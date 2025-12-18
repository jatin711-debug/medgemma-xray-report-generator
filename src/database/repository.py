from fastapi import HTTPException # Changed from http.client
import mysql.connector
from dotenv import load_dotenv
import os
load_dotenv(dotenv_path=".env")

async def store_analysis_result(image_uuid: str, reportPath: str) -> None:
    """
    Store the analysis result in the MySQL database.

    Args:
        image_uuid: Unique identifier for the image.
        reportPath: Path to the generated PDF report.
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

        # Insert the data into the table
        query = "INSERT INTO xray_reports (image_uuid, report_path) VALUES (%s, %s)"
        cursor.execute(query, (image_uuid, reportPath))
        # Commit the transaction
        connection.commit()
        cursor.close()
        connection.close()
    except mysql.connector.Error as err:
        print(f"MySQL Error: {err}")
        # Use FastAPI's HTTPException
        raise HTTPException(status_code=500, detail=f"Failed to store analysis result: {err}")
    except Exception as e:
        print(f"General Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store analysis result: {e}")
    
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
            CREATE TABLE IF NOT EXISTS xray_reports (
                id INT AUTO_INCREMENT PRIMARY KEY,
                image_uuid VARCHAR(255) NOT NULL UNIQUE ,
                report_path VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- Timestamp of when the user was created
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP -- Timestamp of the last update
            );
        """)
        # Create the index separately and handle if it already exists
        try:
            cursor.execute("""
                CREATE INDEX idx_image_uuid ON xray_reports (image_uuid);
            """)
        except mysql.connector.Error as err:
            if err.errno == 1061:  # ER_DUP_KEYNAME (Duplicate key name)
                print(f"Info: Index 'idx_image_uuid' already exists on table 'xray_reports'.")
            else:
                raise # Re-raise other MySQL errors
        
        # It's good practice to close the cursor before committing if no more operations are needed with it.
        cursor.close()
        connection.commit()
        connection.close()
    except mysql.connector.Error as err:
        print(f"MySQL Error during database initialization: {err}")
        # Use FastAPI's HTTPException
        raise HTTPException(status_code=500, detail=f"Failed to initialize database: {err}")
    except Exception as e:
        print(f"General Error during database initialization: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize database: {e}")