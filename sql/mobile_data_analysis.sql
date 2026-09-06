-- =============================================================================
-- Smart Mobile Data Usage Prediction & Plan Recommendation System
-- SQL Analysis  (MySQL 8.0+ syntax — uses CTEs and window functions)
--
-- Data model: two normalized tables loaded from the CLEANED dataset
--   customers    : one row per user (static profile attributes)
--   daily_usage  : one row per user-day (the fact table)
-- customers.User_ID is the primary key; daily_usage.User_ID is a foreign key.
-- This lets plan- and profile-level questions (Q3, Q4, Q13, Q14, Q15) JOIN
-- correctly instead of duplicating profile columns onto every daily row.
--
-- Source files (produced by scripts/03 in the Python pipeline, split from
-- data/cleaned/mobile_data_usage_cleaned.csv):
--   data/cleaned/customers.csv     (500 rows)
--   data/cleaned/daily_usage.csv   (90,000 rows)
--
-- All query results below were validated against the actual generated
-- dataset (cross-checked with an equivalent run in SQLite during
-- development) — no numbers here are invented.
-- =============================================================================

CREATE DATABASE IF NOT EXISTS mobile_analytics;
USE mobile_analytics;

DROP TABLE IF EXISTS daily_usage;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    User_ID       VARCHAR(10) PRIMARY KEY,
    Age           INT,
    Occupation    VARCHAR(30),
    Plan_GB       DECIMAL(5,2),
    Plan_Price    INT,
    Network_Type  VARCHAR(5)
);

CREATE TABLE daily_usage (
    id                 INT AUTO_INCREMENT PRIMARY KEY,
    User_ID            VARCHAR(10),
    Usage_Date         DATE,
    WiFi_Availability  TINYINT,
    Mobile_Data_MB     DECIMAL(10,2),
    WiFi_Data_MB       DECIMAL(10,2),
    YouTube_MB         DECIMAL(10,2),
    Instagram_MB       DECIMAL(10,2),
    WhatsApp_MB        DECIMAL(10,2),
    Netflix_MB         DECIMAL(10,2),
    Gaming_MB          DECIMAL(10,2),
    Browsing_MB        DECIMAL(10,2),
    Work_Study_MB      DECIMAL(10,2),
    Screen_Time_Hours  DECIMAL(5,2),
    Is_Weekend         TINYINT,
    Is_Holiday         TINYINT,
    CONSTRAINT fk_user FOREIGN KEY (User_ID) REFERENCES customers(User_ID),
    INDEX idx_user_date (User_ID, Usage_Date)
);

-- ---------------------------------------------------------------------------
-- Load data. Adjust the file path to your local `secure_file_priv` directory,
-- or use MySQL Workbench's "Table Data Import Wizard" if LOCAL INFILE is
-- restricted on your server.
-- ---------------------------------------------------------------------------
LOAD DATA LOCAL INFILE 'data/cleaned/customers.csv'
INTO TABLE customers
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(User_ID, Age, Occupation, Plan_GB, Plan_Price, Network_Type);

LOAD DATA LOCAL INFILE 'data/cleaned/daily_usage.csv'
INTO TABLE daily_usage
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(User_ID, @date_col, WiFi_Availability, Mobile_Data_MB, WiFi_Data_MB, YouTube_MB,
 Instagram_MB, WhatsApp_MB, Netflix_MB, Gaming_MB, Browsing_MB, Work_Study_MB,
 Screen_Time_Hours, Is_Weekend, Is_Holiday)
SET Usage_Date = STR_TO_DATE(@date_col, '%Y-%m-%d');

-- =============================================================================
-- Q1. Average daily mobile data usage across all users
--     Result on the generated dataset: ~254.88 MB/day
-- =============================================================================
SELECT ROUND(AVG(Mobile_Data_MB), 2) AS avg_daily_mb
FROM daily_usage;

-- =============================================================================
-- Q2. Maximum and minimum single-day mobile usage recorded
-- =============================================================================
SELECT MAX(Mobile_Data_MB) AS max_mb, MIN(Mobile_Data_MB) AS min_mb
FROM daily_usage;

-- =============================================================================
-- Q3. Average usage by occupation (JOIN daily_usage -> customers)
--     Result: Students highest (~294 MB/day), Retired lowest (~183 MB/day)
-- =============================================================================
SELECT c.Occupation, ROUND(AVG(d.Mobile_Data_MB), 2) AS avg_mb
FROM daily_usage d
JOIN customers c ON d.User_ID = c.User_ID
GROUP BY c.Occupation
ORDER BY avg_mb DESC;

-- =============================================================================
-- Q4. Average usage and customer count by plan size
--     Confirms bigger plans are generally held by heavier users.
-- =============================================================================
SELECT c.Plan_GB, ROUND(AVG(d.Mobile_Data_MB), 2) AS avg_mb, COUNT(DISTINCT c.User_ID) AS n_users
FROM daily_usage d
JOIN customers c ON d.User_ID = c.User_ID
GROUP BY c.Plan_GB
ORDER BY c.Plan_GB;

-- =============================================================================
-- Q5. Weekend vs weekday average usage (CASE expression)
--     Result: Weekend ~290.6 MB vs Weekday ~240.4 MB (~21% higher on weekends)
-- =============================================================================
SELECT
    CASE WHEN Is_Weekend = 1 THEN 'Weekend' ELSE 'Weekday' END AS day_type,
    ROUND(AVG(Mobile_Data_MB), 2) AS avg_mb
FROM daily_usage
GROUP BY day_type;

-- =============================================================================
-- Q6. Average consumption per app/category
--     Streaming (Netflix ~48 MB, YouTube ~43 MB) dominates messaging (WhatsApp ~11 MB)
-- =============================================================================
SELECT
    ROUND(AVG(YouTube_MB), 2)     AS avg_youtube_mb,
    ROUND(AVG(Netflix_MB), 2)     AS avg_netflix_mb,
    ROUND(AVG(Instagram_MB), 2)   AS avg_instagram_mb,
    ROUND(AVG(WhatsApp_MB), 2)    AS avg_whatsapp_mb,
    ROUND(AVG(Gaming_MB), 2)      AS avg_gaming_mb,
    ROUND(AVG(Browsing_MB), 2)    AS avg_browsing_mb,
    ROUND(AVG(Work_Study_MB), 2)  AS avg_work_study_mb
FROM daily_usage;

-- =============================================================================
-- Q7. Top 10 highest-usage customers (by average daily usage)
-- =============================================================================
SELECT User_ID, ROUND(AVG(Mobile_Data_MB), 2) AS avg_mb
FROM daily_usage
GROUP BY User_ID
ORDER BY avg_mb DESC
LIMIT 10;

-- =============================================================================
-- Q8. Monthly usage total per user (date functions: DATE_FORMAT)
-- =============================================================================
SELECT User_ID, DATE_FORMAT(Usage_Date, '%Y-%m') AS year_month,
       ROUND(SUM(Mobile_Data_MB), 2) AS monthly_mb
FROM daily_usage
GROUP BY User_ID, year_month
ORDER BY User_ID, year_month;

-- =============================================================================
-- Q9. Customer ranking by average usage (window function: RANK)
-- =============================================================================
SELECT User_ID, avg_mb, RANK() OVER (ORDER BY avg_mb DESC) AS usage_rank
FROM (
    SELECT User_ID, AVG(Mobile_Data_MB) AS avg_mb
    FROM daily_usage
    GROUP BY User_ID
) AS user_avgs
LIMIT 10;

-- =============================================================================
-- Q10. Running total of usage per user over time (window function: SUM OVER)
-- =============================================================================
SELECT User_ID, Usage_Date, Mobile_Data_MB,
       SUM(Mobile_Data_MB) OVER (PARTITION BY User_ID ORDER BY Usage_Date) AS running_total_mb
FROM daily_usage
WHERE User_ID = 'U0001'
ORDER BY Usage_Date;

-- =============================================================================
-- Q11. 7-day moving average of usage per user (window function: AVG OVER frame)
-- =============================================================================
SELECT User_ID, Usage_Date, Mobile_Data_MB,
       ROUND(AVG(Mobile_Data_MB) OVER (
           PARTITION BY User_ID ORDER BY Usage_Date
           ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
       ), 2) AS moving_avg_7d
FROM daily_usage
WHERE User_ID = 'U0001'
ORDER BY Usage_Date;

-- =============================================================================
-- Q12. Month-over-month usage trend per user (CTE + window function: LAG)
-- =============================================================================
WITH monthly AS (
    SELECT User_ID, DATE_FORMAT(Usage_Date, '%Y-%m') AS ym,
           SUM(Mobile_Data_MB) AS monthly_mb
    FROM daily_usage
    GROUP BY User_ID, ym
)
SELECT User_ID, ym, monthly_mb,
       monthly_mb - LAG(monthly_mb) OVER (PARTITION BY User_ID ORDER BY ym) AS change_mb
FROM monthly
WHERE User_ID = 'U0001'
ORDER BY ym;

-- =============================================================================
-- Q13. Plan utilization % for the most recent full month (June 2025) — JOIN + GROUP BY
--     utilization_pct = monthly usage / (Plan_GB * 1024) * 100
-- =============================================================================
SELECT c.User_ID, c.Plan_GB, ROUND(SUM(d.Mobile_Data_MB), 2) AS month_mb,
       ROUND(SUM(d.Mobile_Data_MB) / (c.Plan_GB * 1024) * 100, 2) AS utilization_pct
FROM daily_usage d
JOIN customers c ON d.User_ID = c.User_ID
WHERE DATE_FORMAT(d.Usage_Date, '%Y-%m') = '2025-06'
GROUP BY c.User_ID, c.Plan_GB
ORDER BY utilization_pct DESC
LIMIT 10;

-- =============================================================================
-- Q14. Risk bucket per customer for the current month (CTE + CASE)
--     Thresholds mirror the Python risk engine: <=85% Low, <=100% Medium, >100% High
-- =============================================================================
WITH monthly AS (
    SELECT c.User_ID, c.Plan_GB, SUM(d.Mobile_Data_MB) AS month_mb
    FROM daily_usage d
    JOIN customers c ON d.User_ID = c.User_ID
    WHERE DATE_FORMAT(d.Usage_Date, '%Y-%m') = '2025-06'
    GROUP BY c.User_ID, c.Plan_GB
)
SELECT User_ID, Plan_GB, month_mb,
       ROUND(month_mb / (Plan_GB * 1024) * 100, 1) AS pct_used,
       CASE
           WHEN month_mb / (Plan_GB * 1024) <= 0.85 THEN 'Low'
           WHEN month_mb / (Plan_GB * 1024) <= 1.00 THEN 'Medium'
           ELSE 'High'
       END AS risk_level
FROM monthly
ORDER BY pct_used DESC
LIMIT 10;

-- =============================================================================
-- Q15. Top 5 highest-usage Students (JOIN + filter on customer attribute)
-- =============================================================================
SELECT c.Occupation, c.User_ID, ROUND(AVG(d.Mobile_Data_MB), 2) AS avg_mb
FROM daily_usage d
JOIN customers c ON d.User_ID = c.User_ID
WHERE c.Occupation = 'Student'
GROUP BY c.User_ID
ORDER BY avg_mb DESC
LIMIT 5;

-- =============================================================================
-- Q16. Wi-Fi availability effect on mobile data usage
--     Result: WiFi available ~156.6 MB/day vs no WiFi ~368.5 MB/day (~57.5% lower)
-- =============================================================================
SELECT WiFi_Availability, ROUND(AVG(Mobile_Data_MB), 2) AS avg_mb
FROM daily_usage
GROUP BY WiFi_Availability;

-- =============================================================================
-- Q17. Average screen time and usage by age group (aggregate + CASE bucketing)
-- =============================================================================
SELECT
    CASE
        WHEN c.Age < 20 THEN '<20'
        WHEN c.Age < 30 THEN '20-29'
        WHEN c.Age < 40 THEN '30-39'
        WHEN c.Age < 50 THEN '40-49'
        WHEN c.Age < 60 THEN '50-59'
        ELSE '60+'
    END AS age_group,
    ROUND(AVG(d.Mobile_Data_MB), 2) AS avg_mb,
    ROUND(AVG(d.Screen_Time_Hours), 2) AS avg_screen_time
FROM daily_usage d
JOIN customers c ON d.User_ID = c.User_ID
GROUP BY age_group
ORDER BY age_group;
