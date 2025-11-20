-- 1 --
SELECT * FROM pg_indexes WHERE schemaname = 'my_schema';

-- 2 --
CREATE INDEX idx_cust_city ON cust(city);

EXPLAIN ANALYZE SELECT * FROM cust ORDER BY city;

-- 3 --
INSERT INTO cust (cnum, name, rating, city)
VALUES 
  (generate_series(1, 1000), 
   'test_name', 
   100, 
   'test_city');



WITH sal_orders AS (
    SELECT snum, COUNT(*) as ord_count
    FROM ord
    GROUP BY snum
    HAVING COUNT(*) <= 10
),
moscow_customers AS (
    SELECT MIN(rating) as min_moscow_rating
    FROM cust
    WHERE city = 'Moscow'
)
SELECT 
    o.onum,
    o.amt,
    o.ord_date,
    p.pnum,
    p.name as prod_name,
    p.city as prod_city,
    c.cnum,
    c.name as cust_name,
    c.rating as cust_rating,
    c.city as cust_city,
    s.snum,
    s.name as sal_name,
    s.city as sal_city
FROM ord o
JOIN prod p ON o.pnum = p.pnum
JOIN cust c ON o.cnum = c.cnum
JOIN sal s ON o.snum = s.snum
WHERE 
    o.amt > (SELECT AVG(amt) FROM ord)
    AND p.city != 'Saint Petersburg'
    AND o.snum IN (SELECT snum FROM sal_orders)
    AND c.rating >= (SELECT min_moscow_rating FROM moscow_customers);


-- Таблица --
CREATE TABLE order_info AS
SELECT 
    o.onum,
    o.amt,
    o.ord_date,
    p.pnum,
    p.name as prod_name,
    p.city as prod_city,
    c.cnum,
    c.name as cust_name,
    c.rating as cust_rating,
    c.city as cust_city,
    s.snum,
    s.name as sal_name,
    s.city as sal_city
FROM ord o
JOIN prod p ON o.pnum = p.pnum
JOIN cust c ON o.cnum = c.cnum
JOIN sal s ON o.snum = s.snum
WHERE 
    o.amt > (SELECT AVG(amt) FROM ord)
    AND p.city != 'Saint Petersburg'
    AND o.snum IN (
        SELECT snum 
        FROM ord 
        GROUP BY snum 
        HAVING COUNT(*) <= 10
    )
    AND c.rating >= (
        SELECT MIN(rating) 
        FROM cust 
        WHERE city = 'Moscow'
    );


SELECT * FROM order_info;


-- 5 --
-- Представление --
CREATE VIEW order_info_view AS
SELECT 
    o.onum,
    o.amt,
    o.ord_date,
    p.pnum,
    p.name as prod_name,
    p.city as prod_city,
    c.cnum,
    c.name as cust_name,
    c.rating as cust_rating,
    c.city as cust_city,
    s.snum,
    s.name as sal_name,
    s.city as sal_city
FROM ord o
JOIN prod p ON o.pnum = p.pnum
JOIN cust c ON o.cnum = c.cnum
JOIN sal s ON o.snum = s.snum
WHERE 
    o.amt > (SELECT AVG(amt) FROM ord)
    AND p.city != 'Saint Petersburg'
    AND o.snum IN (
        SELECT snum 
        FROM ord 
        GROUP BY snum 
        HAVING COUNT(*) <= 10
    )
    AND c.rating >= (
        SELECT MIN(rating) 
        FROM cust 
        WHERE city = 'Moscow'
    );

SELECT * FROM order_info_view;


-- 6 --
-- Материализованное представление --

CREATE MATERIALIZED VIEW order_info_materialized AS
SELECT 
    o.onum,
    o.amt,
    o.ord_date,
    p.pnum,
    p.name as prod_name,
    p.city as prod_city,
    c.cnum,
    c.name as cust_name,
    c.rating as cust_rating,
    c.city as cust_city,
    s.snum,
    s.name as sal_name,
    s.city as sal_city
FROM ord o
JOIN prod p ON o.pnum = p.pnum
JOIN cust c ON o.cnum = c.cnum
JOIN sal s ON o.snum = s.snum
WHERE 
    o.amt > (SELECT AVG(amt) FROM ord)
    AND p.city != 'Saint Petersburg'
    AND o.snum IN (
        SELECT snum 
        FROM ord 
        GROUP BY snum 
        HAVING COUNT(*) <= 10
    )
    AND c.rating >= (
        SELECT MIN(rating) 
        FROM cust 
        WHERE city = 'Moscow'
    );

SELECT * FROM order_info_materialized;



-- 7 --
-- блок оператора WITH --
WITH seller_orders AS (
    SELECT snum, COUNT(*) as order_count
    FROM ord
    GROUP BY snum
    HAVING COUNT(*) <= 10
),
moscow_customers AS (
    SELECT MIN(rating) as min_moscow_rating
    FROM cust
    WHERE city = 'Moscow'
),
order_info_result AS (
    SELECT 
        o.onum,
        o.amt,
        o.ord_date,
        p.pnum,
        p.name as product_name,
        p.city as product_city,
        c.cnum,
        c.name as customer_name,
        c.rating as customer_rating,
        c.city as customer_city,
        s.snum,
        s.name as seller_name,
        s.city as seller_city
    FROM ord o
    JOIN prod p ON o.pnum = p.pnum
    JOIN cust c ON o.cnum = c.cnum
    JOIN sal s ON o.snum = s.snum
    WHERE 
        o.amt > (SELECT AVG(amt) FROM ord)
        AND p.city != 'Saint Petersburg'
        AND o.snum IN (SELECT snum FROM seller_orders)
        AND c.rating >= (SELECT min_moscow_rating FROM moscow_customers)
)
SELECT * FROM order_info_result;
