-- using 2 as a seed to the RNG


select cntrycode,
       count(*)       as numcust,
       sum(c_acctbal) as totacctbal
from (select substring(c_phone from 1 for 2) as cntrycode,
             c_acctbal
      from customer
      where substring(c_phone from 1 for 2) in
            ('14', '31', '27', '10', '17', '21', '11')
        and c_acctbal > (select avg(c_acctbal)
                         from customer
                         where c_acctbal > 0.00
                           and substring(c_phone from 1 for 2) in
                               ('14', '31', '27', '10', '17', '21', '11'))
        and not exists(
              select *
              from orders
              where o_custkey = c_custkey
          )) as custsale
group by cntrycode
order by cntrycode;
limit
-1;
