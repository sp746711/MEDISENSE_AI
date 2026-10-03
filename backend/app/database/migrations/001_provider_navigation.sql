-- Migration: 001_provider_navigation.sql
-- Safely add provider navigation and provenance columns to doctors, facilities, and medical_shops tables

-- Doctors table extensions
ALTER TABLE doctors ADD COLUMN IF NOT EXISTS external_id VARCHAR(100);
ALTER TABLE doctors ADD COLUMN IF NOT EXISTS registration_number VARCHAR(100);
ALTER TABLE doctors ADD COLUMN IF NOT EXISTS registration_council VARCHAR(255);
ALTER TABLE doctors ADD COLUMN IF NOT EXISTS latitude DOUBLE PRECISION;
ALTER TABLE doctors ADD COLUMN IF NOT EXISTS longitude DOUBLE PRECISION;
ALTER TABLE doctors ADD COLUMN IF NOT EXISTS contact VARCHAR(100);
ALTER TABLE doctors ADD COLUMN IF NOT EXISTS source_url TEXT;

-- Facilities table extensions
ALTER TABLE facilities ADD COLUMN IF NOT EXISTS latitude DOUBLE PRECISION;
ALTER TABLE facilities ADD COLUMN IF NOT EXISTS longitude DOUBLE PRECISION;
ALTER TABLE facilities ADD COLUMN IF NOT EXISTS contact VARCHAR(100);
ALTER TABLE facilities ADD COLUMN IF NOT EXISTS source_url TEXT;

-- Medical shops table extensions
ALTER TABLE medical_shops ADD COLUMN IF NOT EXISTS latitude DOUBLE PRECISION;
ALTER TABLE medical_shops ADD COLUMN IF NOT EXISTS longitude DOUBLE PRECISION;
ALTER TABLE medical_shops ADD COLUMN IF NOT EXISTS contact VARCHAR(100);
ALTER TABLE medical_shops ADD COLUMN IF NOT EXISTS source_url TEXT;
