# Bikeshare Analytics Transformation

## 1. Project Description

This **dbt project** (`dbt_transform`) transforms raw bikeshare data from snowflake into clean, analytics-ready dimensional models. The project follows a **modular data modeling approach** using dbt’s *staging → intermediate → marts* structure to ensure scalability, maintainability, and performance.

The overall goal is to transform and model bikeshare ride and station datasets into dimension and fact tables optimised for downstream analytics, dashboards, and reporting.

---

## 2. Project Architecture

### 🧩 2.1 Model Layers

#### **1. Staging Layer (`models/staging/`)**

* **Purpose:** Cleans and standardizes raw source data from the `raw` schema.
* **Models:**

  * `stg_rides.sql`: Cleans and formats ride data (standardizes field names, timestamps, and user types).
  * `stg_stations_current.sql`: Processes current station metadata.
* **Key Features:**

  * Source freshness validation using `sources.yml`
  * Type casting and column renaming for consistency
  * Provides a single source of truth for downstream transformations

---

#### **2. Intermediate Layer (`models/intermediate/`)**

* **Purpose:** Serves as a transformation zone between staging and the final data marts.
* **Structure:**

  * `dims/` → Builds dimension tables used for joining and analysis
  * `fct/` → Builds fact tables representing measurable business events

##### **Dimension Models (`intermediate/dims/`)**

| Model                   | Description                                       | Type                                            |
| ----------------------- | ------------------------------------------------- | ----------------------------------------------- |
| `dim_date.sql`          | Date dimension table for calendar-based joins     | Type 1                                          |
| `dim_rideable_type.sql` | Categorical lookup for rideable types             | Type 1                                          |
| `dim_station.sql`       | Station metadata dimension with snapshot tracking | Slowly Changing Dimension (Type 2 via snapshot) |
| `dim_user_type.sql`     | User classification (member vs casual)            | Type 1                                          |

##### **Fact Models (`intermediate/fct/`)**

| Model           | Description                                                                                                                |
| --------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `fct_rides.sql` | Central fact table capturing each ride event with foreign keys to dimensions (station, user type, rideable type, and date) |

---

#### **3. Snapshots Layer (`snapshots/`)**

* **Purpose:** Tracks historical changes in slowly changing dimensions (SCD Type 2).
* **Model:**

  * `dim_station_snapshot.sql`: Captures station-level attribute changes (e.g., name or location updates) over time using dbt’s `snapshot` functionality.
* **Logic:** Uses `dbt_valid_from` and `dbt_valid_to` fields to track when each version of a station record was valid.

---

## 3. Incremental Strategies

Where necessary, incremental models are used to optimize performance and minimize compute costs:

* **`fct_rides.sql`** uses *incremental loading* to append only new rides (`ride_id` not already present in the existing table).
* Merge-based logic (supported by Snowflake/BigQuery) ensures updates and inserts happen in one efficient command.
* This design supports scalable daily loads without reprocessing historical data.

---

## 4. Data Modeling Approach

The project follows a **Kimball-style star schema**:

* **Fact Table:** `fct_rides`
* **Dimension Tables:** `dim_date`, `dim_user_type`, `dim_rideable_type`, `dim_station`

This structure simplifies analytical queries and ensures flexibility for aggregation across multiple dimensions (e.g., rides by user type, station, or date).

---

## 5. Testing and Documentation

* **Schema Tests:** Implemented in `schema.yml` to enforce:

  * Uniqueness and non-null constraints on primary keys (e.g., `ride_id`, `station_id`)
  * Referential integrity between fact and dimension tables
* **Source Tests:** Ensure completeness and consistency of raw data.
* **Documentation:** Each model includes `description` fields for automatic dbt documentation generation (`dbt docs generate`).

---

## 6. Orchestration and Deployment

* Models are materialized as **incremental**, **views**, or **tables** depending on usage frequency and data volume.
* The project can be scheduled using an orchestrator (e.g., Airflow or dbt Cloud scheduler).
* The transformation pipeline supports CI/CD through version control and automated tests.

---

## 7. Future Enhancements

* Add **dbt exposures** to link models to downstream dashboards.
* Introduce **data quality alerts** using dbt metrics or external observability tools.
* Extend to **SCD Type 2 handling** for user or rideable dimensions if attributes become mutable.
* Integrate **semantic models** for BI layer alignment.

---

## 8. Summary

| Layer        | Purpose                           | Example Models                                  |
| ------------ | --------------------------------- | ----------------------------------------------- |
| Staging      | Standardize and clean raw data    | `stg_rides`, `stg_stations_current`             |
| Intermediate | Transform and enrich              | `dim_station`, `dim_user_type`, `fct_rides`     |
| Snapshots    | Track history                     | `dim_station_snapshot`                          |
| Marts        | (Optional) Serve analytical needs | Could be added later for dashboard aggregations |
