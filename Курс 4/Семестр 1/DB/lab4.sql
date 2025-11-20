set search_path to my_schema;

-- 1 --
select * from sal
where comm >= (select avg(comm) from sal);

-- 2 --
select cnum, sum(amt) from ord
group by cnum
having sum(amt) <= (
  select sum(amt) from ord where cnum = 2003
);

-- 3 --
select snum from ord
where pnum = 1001 and cnum in (
  select cnum from ord where pnum = 1005
);

-- 4 --
select name from prod
where pnum not in (
  select pnum from ord where snum in (
    select snum from sal where city = 'Moscow'
  )
);

-- 5 --
select name from cust
where rating >= any (
  select rating from cust where city = 'Moscow'
);

-- 6 --
select name from sal
where snum not in (
  select snum from ord
  where cnum in (
    select cnum from cust
    where city in (
      select city from sal
      where sal.snum = ord.snum
    )
  )
);

-- 7 --
select snum, name, comm, city,
    case
        when comm < 0.12 then 'low'
        when comm >= 0.12 and comm <= 0.13 then 'medium'
        when comm > 0.13 then 'high'
    end as "commission_level"
from sal;

-- 8 --
select snum, count(*) as order_count,
    case
        when count(*) >= 4 then 'active'
        when count(*) between 2 and 3 then 'moderate'
        when count(*) = 1 then 'inactive'
    end as activity_level
from ord
group by snum;
