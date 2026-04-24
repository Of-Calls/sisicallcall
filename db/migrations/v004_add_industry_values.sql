-- industry 체크 제약 확장 — government(구청/관공서), retail(소매점) 추가
-- 기존: hospital | restaurant | finance | appliance
-- 변경: hospital | restaurant | finance | appliance | government | retail

ALTER TABLE tenants
    DROP CONSTRAINT tenants_industry_check;

ALTER TABLE tenants
    ADD CONSTRAINT tenants_industry_check
        CHECK (industry IN (
            'hospital',
            'restaurant',
            'finance',
            'appliance',
            'government',
            'retail'
        ));
