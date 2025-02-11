# backend/fair_eval_backend/data_processing/pipeline.py

import psycopg2
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
    df = map_departments(df)
    df = preprocess_comments(df)

    print("\n--- Stage 3: Database Population ---")
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()

    try:
        # 1. Insert departments with batch processing
        dept_data = df[['department', 'discipline', 'sub_discipline']].drop_duplicates()
        insert_departments_sql = """
            INSERT INTO api_department (name, discipline, sub_discipline)
            VALUES (%s, %s, %s)
            ON CONFLICT (name) DO NOTHING
        """
        cursor.executemany(insert_departments_sql, dept_data.values.tolist())
        conn.commit()

        # 2. Insert professors with batch processing (updated to include would_take_again_percent)
        prof_data = df[['professor_id', 'first_name', 'last_name', 'gender', 'department', 'would_take_again_percent']].drop_duplicates()
        insert_professors_sql = """
            WITH dept_id AS (
                SELECT id FROM api_department WHERE name = %s
            )
            INSERT INTO api_professor (
                professor_id, first_name, last_name, gender, 
                 would_take_again_percent, department_id
            )
            VALUES (%s, %s, %s, %s, (SELECT id FROM dept_id), %s)
            ON CONFLICT (professor_id) DO NOTHING
        """
        cursor.executemany(insert_professors_sql, 
                          [(row['department'], row['professor_id'], row['first_name'], 
                            row['last_name'], row['gender'], row['would_take_again_percent']) 
                           for _, row in prof_data.iterrows()])
        conn.commit()

        # 2.1 Create users for professors
        insert_users_sql = """
            INSERT INTO auth_user (
                username, first_name, last_name, email,
                password, is_staff, is_active, date_joined
            )
            SELECT 
                p.professor_id,  -- username
                p.first_name,
                p.last_name,
                LOWER(CONCAT(p.first_name, '.', p.last_name, '@institution.edu')),  -- email
                'pbkdf2_sha256$600000$default_hash',  -- default hashed password
                TRUE,  -- is_staff
                TRUE,  -- is_active
                NOW()  -- date_joined
            FROM api_professor p
            LEFT JOIN auth_user u ON u.username = p.professor_id
            WHERE u.id IS NULL
            RETURNING id, username
        """
        cursor.execute(insert_users_sql)
        user_mappings = cursor.fetchall()
        conn.commit()

        # 2.2 Assign roles (20% Admin, 80% Academic)
        admin_count = int(len(user_mappings) * 0.2)
        insert_roles_sql = """
            INSERT INTO api_userrole (user_id, role, department_id)
            SELECT 
                u.id,
                CASE 
                    WHEN ROW_NUMBER() OVER (ORDER BY RANDOM()) <= %s THEN 'ADMIN'
                    ELSE 'ACADEMIC'
                END,
                p.department_id
            FROM auth_user u
            JOIN api_professor p ON p.professor_id = u.username
            LEFT JOIN api_userrole ur ON ur.user_id = u.id
            WHERE ur.id IS NULL
        """
        cursor.execute(insert_roles_sql, (admin_count,))
        conn.commit()

        # 3. Modified ratings insertion with correct schema
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

        # 4. Add sentiment table insertion
        insert_sentiment_sql = """
            INSERT INTO api_sentiment (
                professor_id, comment, processed_comment,
                sentiment, created_at
            )
            SELECT 
                p.id,
                s.comment,
                s.proc_comment,
                NULL,  -- sentiment will be computed later
                s.created_at
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
            flag_statuses = chunk['rating_flagstatus'].tolist()
            helpful_ratings = chunk['rating_helpfulrating'].tolist()
            clarity_ratings = chunk['rating_clarityrating'].tolist()
            difficulty_ratings = chunk['rating_difficultyrating'].tolist()
            is_onlines = chunk['rating_isforonlineclass'].tolist()
            is_for_credits = chunk['rating_isforcredit'].tolist()
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