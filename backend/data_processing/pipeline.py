# backend/fair_eval_backend/data_processing/pipeline.py

import psycopg2
import pandas as pd
from datetime import datetime
from data_processing.ingestion import ingest_csv_to_df
from data_processing.cleaning import clean_data
from data_processing.gender_assignment import engineer_gender
from data_processing.dept_mapping import map_departments
from data_processing.text_preprocessing import preprocess_comments

def run_full_pipeline(csv_path, db_config):
    """Optimized pipeline with batch processing"""
    print("\n--- Stage 1: Data Ingestion ---")
    df = ingest_csv_to_df(csv_path)
    
    print("\n--- Stage 2: Data Processing ---")
    df = clean_data(df)
    df = engineer_gender(df)
    df = map_departments(df)  # This now returns discipline and sub_discipline
    df = preprocess_comments(df)
    
    # Save processed data after Stage 2
    processed_csv_path = csv_path.replace('.csv', '_processed.csv')
    df.to_csv(processed_csv_path, index=False)
    print(f"\nProcessed data saved to: {processed_csv_path}")

    run_db_population(processed_csv_path, db_config)

def run_db_population(processed_csv_path, db_config):
    """Run only the database population stage using processed CSV"""
    print("\n--- Loading processed data ---")
    df = pd.read_csv(processed_csv_path)
    
    print("\n--- Stage 3: Database Population ---")
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    
    try:
        # 1. Insert professors with discipline information
        prof_data = df[['professor_id', 'first_name', 'last_name', 'gender', 
                        'discipline', 'sub_discipline']].drop_duplicates()

        insert_professors_sql = """
            INSERT INTO api_professor (
                professor_id, first_name, last_name, gender,
                discipline, sub_discipline
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (professor_id) DO NOTHING
        """
        cursor.executemany(insert_professors_sql, 
                          [(row['professor_id'], row['first_name'], row['last_name'],
                            row['gender'], row['discipline'], row['sub_discipline']) 
                           for _, row in prof_data.iterrows()])
        conn.commit()

        # 2. Create users for professors
        insert_users_sql = """
            INSERT INTO auth_user (
                username, first_name, last_name, email,
                password, is_staff, is_active, date_joined,
                is_superuser
            )
            SELECT 
                p.professor_id,
                p.first_name,
                p.last_name,
                LOWER(CONCAT(p.first_name, '.', p.last_name, '@institution.edu')),
                'pbkdf2_sha256$600000$default_hash',
                TRUE,
                TRUE,
                NOW(),
                FALSE
            FROM api_professor p
            LEFT JOIN auth_user u ON u.username = p.professor_id
            WHERE u.id IS NULL
            RETURNING id, username
        """
        cursor.execute(insert_users_sql)
        user_mappings = cursor.fetchall()
        conn.commit()

        # 2.1 Assign roles (20% Admin, 80% Academic)
        admin_count = int(len(user_mappings) * 0.2)
        insert_roles_sql = """
            INSERT INTO api_userrole (user_id, role, discipline)
            SELECT 
                u.id,
                CASE 
                    WHEN ROW_NUMBER() OVER (ORDER BY RANDOM()) <= %s THEN 'ADMIN'
                    ELSE 'ACADEMIC'
                END,
                p.discipline  -- Get discipline directly from professor table
            FROM auth_user u
            JOIN api_professor p ON p.professor_id = u.username
            LEFT JOIN api_userrole ur ON ur.user_id = u.id
            WHERE ur.id IS NULL
        """
        cursor.execute(insert_roles_sql, (admin_count,))
        conn.commit()

        # 3. Insert ratings
        batch_size = 5000
        insert_ratings_sql = """
            INSERT INTO api_rating (
                professor_id, avg_rating, flag_status, helpful_rating,
                clarity_rating, difficulty_rating, is_online,
                is_for_credit, created_at
            )
            SELECT 
                p.id,
                r.avg_rating,
                r.flag_status,
                r.helpful_rating,
                r.clarity_rating,
                r.difficulty_rating,
                r.is_online,
                r.is_for_credit,
                r.created_at
            FROM (
                SELECT * FROM unnest(
                    %s::text[],       -- professor_id
                    %s::float[],      -- avg_rating
                    %s::text[],       -- flag_status
                    %s::float[],      -- helpful_rating
                    %s::float[],      -- clarity_rating
                    %s::float[],      -- difficulty_rating
                    %s::boolean[],    -- is_online
                    %s::boolean[],    -- is_for_credit
                    %s::timestamp[]   -- created_at
                ) AS r(
                    prof_id, avg_rating, flag_status, helpful_rating,
                    clarity_rating, difficulty_rating, is_online,
                    is_for_credit, created_at
                )
            ) r
            JOIN api_professor p ON p.professor_id = r.prof_id
        """

        # 4. Insert sentiments
        insert_sentiment_sql = """
            INSERT INTO api_sentiment (
                professor_id, comment, processed_comment,
                sentiment, created_at, 
                positive_terms, negative_terms,
                vader_compound, vader_positive,
                vader_negative, vader_neutral
            )
            SELECT 
                p.id,
                s.comment,
                s.proc_comment,
                NULL,  -- sentiment will be computed later
                s.created_at,
                NULL, NULL,  -- positive/negative terms
                NULL, NULL, NULL, NULL  -- VADER scores
            FROM (
                SELECT * FROM unnest(
                    %s::text[],       -- professor_id
                    %s::text[],       -- comment
                    %s::text[],       -- processed_comment
                    %s::timestamp[]   -- created_at
                ) AS s(
                    prof_id, comment, proc_comment, created_at
                )
            ) s
            JOIN api_professor p ON p.professor_id = s.prof_id
        """
        
        current_time = datetime.now()
        rating_chunks = [df[i:i + batch_size] for i in range(0, len(df), batch_size)]
        for chunk in rating_chunks:
            # Prepare data arrays for ratings
            prof_ids = chunk['professor_id'].tolist()
            avg_ratings = chunk['avg_rating'].tolist()
            flag_statuses = chunk['flag_status'].tolist()
            helpful_ratings = chunk['helpful_rating'].tolist()
            clarity_ratings = chunk['clarity_rating'].tolist()
            difficulty_ratings = chunk['difficulty_rating'].tolist()
            is_onlines = chunk['is_online'].tolist()
            is_for_credits = chunk['is_for_credit'].tolist()
            timestamps = [current_time] * len(chunk)
            
            # Execute ratings insertion
            cursor.execute(insert_ratings_sql, (
                prof_ids, avg_ratings, flag_statuses, helpful_ratings,
                clarity_ratings, difficulty_ratings, is_onlines,
                is_for_credits, timestamps
            ))

            # Execute sentiment insertion
            cursor.execute(insert_sentiment_sql, (
                prof_ids,
                chunk['rating_comment'].tolist(),
                chunk['processed_comment'].tolist(),
                timestamps
            ))
            
            conn.commit()

    except Exception as e:
        conn.rollback()
        print(f"Error during database population: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()
        print("Pipeline complete! Data saved to normalized tables.")