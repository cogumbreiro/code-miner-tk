DROP TABLE IF EXISTS state_confidence;
CREATE TABLE IF NOT EXISTS state_confidence (
    call TEXT,
    state_var INT,
    symbol TEXT,
    score REAL,
    count INT
);

INSERT INTO state_confidence(call, state_var, symbol, score, count)

SELECT
    L.call,
    L.state_var,
    '0',
    R.count /CAST(L.count as REAL),
    R.count
FROM (
    SELECT call,state_var,count(*) as count
    FROM anomalies
    GROUP by call,state_var
) as L
JOIN (
    SELECT call,state_var,count(*) as count
    FROM anomalies
    WHERE symbol='0'
    GROUP BY call,state_var
) as R
ON L.call = R.call
AND L.state_var = R.state_var
;

INSERT INTO state_confidence(call, state_var, symbol, score, count)
SELECT
    L.call,
    L.state_var,
    '1',
    R.count /CAST(L.count as REAL),
    R.count
FROM (
    SELECT call,state_var,count(*) as count
    FROM anomalies
    GROUP by call,state_var
) as L
JOIN (
    SELECT call,state_var,count(*) as count
    FROM anomalies
    WHERE symbol='1'
    GROUP BY call,state_var
) as R
ON L.call = R.call
AND L.state_var = R.state_var
;
