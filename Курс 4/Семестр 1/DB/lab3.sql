set search_path to my_schema;

-- 1 --
select count(name) from prod
where length(name) > 7;

-- 2 --
select min(comm), max(comm) from sal;

-- 3 --
select snum, min(pnum) from ord
group by snum;

-- 4 --
select count(distinct pnum) from ord
where snum >= 3001 and snum <= 3004;

-- 5 --
select pnum, sum(amt) from ord
group by pnum
having sum(amt) >= 15;

-- 6 --
select pnum, cnum, count(onum) from ord
group by pnum, cnum;

-- 7 --
select ord_date from ord
group by ord_date
having count(distinct snum) = 2
and count(distinct pnum) > 1;
