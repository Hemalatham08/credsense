-- ============================================================
-- CardioSense Database Schema
-- No auth tables yet (login is a future enhancement)
-- Designed to match the exact ML feature set from train_model.py:
-- age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc,
-- smoke, alco, active, bmi, pulse_pressure, map, bp_category
-- ============================================================

-- ---------------------------------------------------------------
-- PATIENTS
-- ---------------------------------------------------------------
CREATE TABLE patients (
    id              SERIAL PRIMARY KEY,
    patient_code    VARCHAR(20) UNIQUE NOT NULL,   -- human-friendly ID, e.g. "PT-0001"
    name            VARCHAR(100) NOT NULL,
    gender          SMALLINT NOT NULL CHECK (gender IN (0, 1)),  -- 0=female, 1=male (matches model encoding)
    date_of_birth   DATE NOT NULL,
    phone           VARCHAR(20),
    email           VARCHAR(100),
    address         TEXT,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- ---------------------------------------------------------------
-- ASSESSMENTS  (one row per clinical visit / risk check)
-- ---------------------------------------------------------------
CREATE TABLE assessments (
    id                SERIAL PRIMARY KEY,
    patient_id        INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    assessment_date   TIMESTAMP DEFAULT NOW(),

    -- Vitals (feed the model directly)
    height            NUMERIC(5,2) NOT NULL,   -- cm
    weight            NUMERIC(5,2) NOT NULL,   -- kg
    ap_hi             SMALLINT NOT NULL,        -- systolic BP
    ap_lo             SMALLINT NOT NULL,        -- diastolic BP

    -- Engineered features (computed by backend at insert time, stored for audit/history)
    bmi               NUMERIC(5,2),
    pulse_pressure    SMALLINT,
    map               NUMERIC(5,2),
    bp_category       SMALLINT,                 -- 0=normal,1=elevated,2=stage1,3=stage2

    created_at        TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_assessments_patient ON assessments(patient_id);

-- ---------------------------------------------------------------
-- LAB RESULTS
-- ---------------------------------------------------------------
CREATE TABLE lab_results (
    id              SERIAL PRIMARY KEY,
    assessment_id   INTEGER NOT NULL REFERENCES assessments(id) ON DELETE CASCADE,
    cholesterol     SMALLINT NOT NULL CHECK (cholesterol IN (1,2,3)),  -- 1=normal,2=above,3=well above
    gluc            SMALLINT NOT NULL CHECK (gluc IN (1,2,3)),         -- 1=normal,2=above,3=well above
    created_at      TIMESTAMP DEFAULT NOW()
);

-- ---------------------------------------------------------------
-- LIFESTYLE FACTORS
-- ---------------------------------------------------------------
CREATE TABLE lifestyle_factors (
    id              SERIAL PRIMARY KEY,
    assessment_id   INTEGER NOT NULL REFERENCES assessments(id) ON DELETE CASCADE,
    smoke           SMALLINT NOT NULL CHECK (smoke IN (0,1)),
    alco            SMALLINT NOT NULL CHECK (alco IN (0,1)),
    active          SMALLINT NOT NULL CHECK (active IN (0,1)),
    created_at      TIMESTAMP DEFAULT NOW()
);

-- ---------------------------------------------------------------
-- PREDICTIONS  (model output for a given assessment)
-- ---------------------------------------------------------------
CREATE TABLE predictions (
    id                  SERIAL PRIMARY KEY,
    assessment_id       INTEGER NOT NULL REFERENCES assessments(id) ON DELETE CASCADE,
    model_used          VARCHAR(50) DEFAULT 'voting_ensemble',
    risk_probability    NUMERIC(5,4) NOT NULL,   -- 0.0000 - 1.0000
    risk_level          VARCHAR(20) NOT NULL,    -- 'Low' / 'Moderate' / 'High'
    confidence_score    NUMERIC(5,4),
    shap_values         JSONB,                    -- per-feature contribution, for the explainability UI
    created_at          TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_predictions_assessment ON predictions(assessment_id);

-- ---------------------------------------------------------------
-- RECOMMENDATIONS  (generated from risk level + top SHAP drivers)
-- ---------------------------------------------------------------
CREATE TABLE recommendations (
    id                  SERIAL PRIMARY KEY,
    prediction_id       INTEGER NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    recommendation_text TEXT NOT NULL,
    priority            VARCHAR(10) DEFAULT 'medium',  -- low/medium/high
    created_at          TIMESTAMP DEFAULT NOW()
);
