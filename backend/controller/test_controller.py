from sqlalchemy import text
from database import engine
from quick_load import import_all_csvs
from sqlalchemy.exc import SQLAlchemyError
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
def get_customers(i=1):

    try:
        with engine.connect() as connection:

            result = connection.execute(
                text("""
                    SELECT *
                    FROM dim_customers
                    LIMIT 100
                """)
            )

            rows = result.fetchall()
            logger.info(f"Data number --target-- : {len(rows)}")
            
            
            return [dict(row._mapping) for row in rows]


    except SQLAlchemyError as e:

        print(f"Database error: {e}")

        logger.info(f"THIS ATTEMPT FAILED SO TRY AGAIN IN ERROR.")
        
        if i > 0:
            print("Reloading data...")
            import_all_csvs()
            return get_customers(i-1)

        return []
    except Exception as e:
        logger.info(f"THIS ATTEMPT FAILED SO TRY AGAIN IN ERROR. heres why : {e} ")