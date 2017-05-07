### Mode Analytics Case Study _ Mahesh Yerra

#### Investigating Drop in User Engagement:

#### Hypothesis:
1- large number employees may be on vacation: 
	We don't have leave data of employees or users to test this hypothesis

2- Application issues
3- Motivational indicator - could be an indicator for  possible rise in attrition rate in near future.

##### Number of Active Vs Pending Users from user tabel
SELECT state, count(user_id) AS user_count
FROM tutorial.yammer_users GROUP BY state

--active = 9381
--pending = 9685

##### Investigating spike in user_engagement in the week 07/28:

SELECT events.interval,users.state,count(DISTINCT users.user_id)
FROM 
(SELECT *,
 CASE WHEN occurred_at::timestamp >= '2014-04-21' 
      AND occurred_at::timestamp < '2014-04-28' THEN 'week_0421'
      WHEN occurred_at::timestamp >= '2014-04-28' 
      AND occurred_at::timestamp < '2014-05-05' THEN 'week_0428'
      WHEN occurred_at::timestamp >= '2014-05-05' 
      AND occurred_at::timestamp < '2014-05-12' THEN 'week_0505'
      WHEN occurred_at::timestamp >= '2014-05-12' 
      AND occurred_at::timestamp < '2014-05-19' THEN 'week_0512'
      WHEN occurred_at::timestamp >= '2014-05-19' 
      AND occurred_at::timestamp < '2014-05-26' THEN 'week_0519'
      WHEN occurred_at::timestamp >= '2014-05-26' 
      AND occurred_at::timestamp < '2014-06-02' THEN 'week_0526'
      WHEN occurred_at::timestamp >= '2014-06-02' 
      AND occurred_at::timestamp < '2014-06-09' THEN 'week_0602'
      WHEN occurred_at::timestamp >= '2014-06-09' 
      AND occurred_at::timestamp < '2014-06-16' THEN 'week_0609'
      WHEN occurred_at::timestamp >= '2014-06-16' 
      AND occurred_at::timestamp < '2014-06-23' THEN 'week_0616'
      WHEN occurred_at::timestamp >= '2014-06-23' 
      AND occurred_at::timestamp < '2014-06-30' THEN 'week_0623'
      WHEN occurred_at::timestamp >= '2014-06-30' 
      AND occurred_at::timestamp < '2014-07-07' THEN 'week_0630'
      WHEN occurred_at::timestamp >= '2014-07-07' 
      AND occurred_at::timestamp < '2014-07-14' THEN 'week_0707'
      WHEN occurred_at::timestamp >= '2014-07-14' 
      AND occurred_at::timestamp < '2014-07-21' THEN 'week_0714'
      WHEN occurred_at::timestamp >= '2014-07-21' 
      AND occurred_at::timestamp < '2014-07-28' THEN 'week_0721'
      WHEN occurred_at::timestamp >= '2014-07-28 23:59:59' 
      AND occurred_at::timestamp < '2014-08-04' THEN 'week_0728'
      WHEN occurred_at::timestamp >= '2014-08-04' 
      AND occurred_at::timestamp < '2014-08-11' THEN 'week_0804'
      WHEN occurred_at::timestamp >= '2014-08-11' 
      AND occurred_at::timestamp < '2014-08-18' THEN 'week_0811'
      WHEN occurred_at::timestamp >= '2014-08-18' 
      AND occurred_at::timestamp < '2014-08-25' THEN 'week_0818'
      WHEN occurred_at::timestamp >= '2014-08-25' 
      AND occurred_at::timestamp < '2014-09-01' THEN 'week_0825'
      WHEN occurred_at::timestamp >= '2014-09-01' 
      AND occurred_at::timestamp < '2014-09-08' THEN 'week_0901'
      ELSE 'out of range' END AS interval
FROM tutorial.yammer_events) events
LEFT JOIN tutorial.yammer_users users 
ON events.user_id = users.user_id
GROUP BY 1,2 ORDER BY 1,2

####### Conclusion:
--conlusion from above query actual  number of active users for  week 0728 is only  1317, However the chart shows  1442
-- This could be due to errors in manual processing while generating reports and charts for this week.


##### investigating  tenure wise active user counts
SELECT * FROM
(
SELECT sub.event_type,sub.interval,
  CASE 
   WHEN sub.user_wks > 0 AND sub.user_wks <= 1 THEN '1st wk'
   WHEN sub.user_wks > 1 AND sub.user_wks <= 2 THEN '2nd wk'
   WHEN sub.user_wks > 2 AND sub.user_wks <= 3 THEN '3rd wk'
   WHEN sub.user_wks > 3 AND sub.user_wks <= 4 THEN '4th wk'
   WHEN sub.user_wks > 4 AND sub.user_wks <= 5 THEN '5th wk'
   WHEN sub.user_wks > 5 AND sub.user_wks <= 6 THEN '6th wk'
   WHEN sub.user_wks > 6 AND sub.user_wks <= 7 THEN '7th wk'
   WHEN sub.user_wks > 7 AND sub.user_wks <= 10 THEN '7 - 10 wks'
   WHEN sub.user_wks > 10 AND sub.user_wks <= 12 THEN '10 - 12 wks'
   WHEN sub.user_wks > 12 AND sub.user_wks <= 16 THEN '12 - 16 wks'
   WHEN sub.user_wks > 16 AND sub.user_wks <= 20 THEN '16 - 20 wks'
   ELSE '20+ wks' END AS user_tenure,
  COUNT(DISTINCT sub.uid)
FROM
(SELECT *, events.user_id AS uid,
(EXTRACT(year FROM (events.occurred_at::timestamp - users.created_at::timestamp))*365+
 EXTRACT(month FROM (events.occurred_at::timestamp - users.created_at::timestamp))*30+
 EXTRACT(day FROM (events.occurred_at::timestamp - users.created_at::timestamp)))/7 AS user_wks
FROM 
(SELECT *,
 CASE WHEN occurred_at::timestamp >= '2014-04-21' 
      AND occurred_at::timestamp < '2014-04-28' THEN 'week_0421'
      WHEN occurred_at::timestamp >= '2014-04-28' 
      AND occurred_at::timestamp < '2014-05-05' THEN 'week_0428'
      WHEN occurred_at::timestamp >= '2014-05-05' 
      AND occurred_at::timestamp < '2014-05-12' THEN 'week_0505'
      WHEN occurred_at::timestamp >= '2014-05-12' 
      AND occurred_at::timestamp < '2014-05-19' THEN 'week_0512'
      WHEN occurred_at::timestamp >= '2014-05-19' 
      AND occurred_at::timestamp < '2014-05-26' THEN 'week_0519'
      WHEN occurred_at::timestamp >= '2014-05-26' 
      AND occurred_at::timestamp < '2014-06-02' THEN 'week_0526'
      WHEN occurred_at::timestamp >= '2014-06-02' 
      AND occurred_at::timestamp < '2014-06-09' THEN 'week_0602'
      WHEN occurred_at::timestamp >= '2014-06-09' 
      AND occurred_at::timestamp < '2014-06-16' THEN 'week_0609'
      WHEN occurred_at::timestamp >= '2014-06-16' 
      AND occurred_at::timestamp < '2014-06-23' THEN 'week_0616'
      WHEN occurred_at::timestamp >= '2014-06-23' 
      AND occurred_at::timestamp < '2014-06-30' THEN 'week_0623'
      WHEN occurred_at::timestamp >= '2014-06-30' 
      AND occurred_at::timestamp < '2014-07-07' THEN 'week_0630'
      WHEN occurred_at::timestamp >= '2014-07-07' 
      AND occurred_at::timestamp < '2014-07-14' THEN 'week_0707'
      WHEN occurred_at::timestamp >= '2014-07-14' 
      AND occurred_at::timestamp < '2014-07-21' THEN 'week_0714'
      WHEN occurred_at::timestamp >= '2014-07-21' 
      AND occurred_at::timestamp < '2014-07-28' THEN 'week_0721'
      WHEN occurred_at::timestamp >= '2014-07-28 23:59:59' 
      AND occurred_at::timestamp < '2014-08-04' THEN 'week_0728'
      WHEN occurred_at::timestamp >= '2014-08-04' 
      AND occurred_at::timestamp < '2014-08-11' THEN 'week_0804'
      WHEN occurred_at::timestamp >= '2014-08-11' 
      AND occurred_at::timestamp < '2014-08-18' THEN 'week_0811'
      WHEN occurred_at::timestamp >= '2014-08-18' 
      AND occurred_at::timestamp < '2014-08-25' THEN 'week_0818'
      WHEN occurred_at::timestamp >= '2014-08-25' 
      AND occurred_at::timestamp < '2014-09-01' THEN 'week_0825'
      WHEN occurred_at::timestamp >= '2014-09-01' 
      AND occurred_at::timestamp < '2014-09-08' THEN 'week_0901'
      ELSE 'out of range' END AS interval
FROM tutorial.yammer_events) events
LEFT JOIN tutorial.yammer_users users
ON events.user_id = users.user_id
)sub
WHERE sub.state = 'active'
GROUP BY 1,2,3 ORDER BY 1,2,4 DESC
)x
WHERE x.user_tenure = '20+ wks'

####### Conclusion:
-- it is observed that users with 20+ weeks tenure followed by  users with 1-3 wks tenure have highest engagement level.  it looks like engagement  at all tenure levels is dipping down after 07-28.

##### Investigating device level data:

SELECT device,interval,COUNT(DISTINCT uid)
FROM 
(SELECT *,user_id AS uid,
 CASE WHEN occurred_at::timestamp >= '2014-04-21' 
      AND occurred_at::timestamp < '2014-04-28' THEN 'week_0421'
      WHEN occurred_at::timestamp >= '2014-04-28' 
      AND occurred_at::timestamp < '2014-05-05' THEN 'week_0428'
      WHEN occurred_at::timestamp >= '2014-05-05' 
      AND occurred_at::timestamp < '2014-05-12' THEN 'week_0505'
      WHEN occurred_at::timestamp >= '2014-05-12' 
      AND occurred_at::timestamp < '2014-05-19' THEN 'week_0512'
      WHEN occurred_at::timestamp >= '2014-05-19' 
      AND occurred_at::timestamp < '2014-05-26' THEN 'week_0519'
      WHEN occurred_at::timestamp >= '2014-05-26' 
      AND occurred_at::timestamp < '2014-06-02' THEN 'week_0526'
      WHEN occurred_at::timestamp >= '2014-06-02' 
      AND occurred_at::timestamp < '2014-06-09' THEN 'week_0602'
      WHEN occurred_at::timestamp >= '2014-06-09' 
      AND occurred_at::timestamp < '2014-06-16' THEN 'week_0609'
      WHEN occurred_at::timestamp >= '2014-06-16' 
      AND occurred_at::timestamp < '2014-06-23' THEN 'week_0616'
      WHEN occurred_at::timestamp >= '2014-06-23' 
      AND occurred_at::timestamp < '2014-06-30' THEN 'week_0623'
      WHEN occurred_at::timestamp >= '2014-06-30' 
      AND occurred_at::timestamp < '2014-07-07' THEN 'week_0630'
      WHEN occurred_at::timestamp >= '2014-07-07' 
      AND occurred_at::timestamp < '2014-07-14' THEN 'week_0707'
      WHEN occurred_at::timestamp >= '2014-07-14' 
      AND occurred_at::timestamp < '2014-07-21' THEN 'week_0714'
      WHEN occurred_at::timestamp >= '2014-07-21' 
      AND occurred_at::timestamp < '2014-07-28' THEN 'week_0721'
      WHEN occurred_at::timestamp >= '2014-07-28 23:59:59' 
      AND occurred_at::timestamp < '2014-08-04' THEN 'week_0728'
      WHEN occurred_at::timestamp >= '2014-08-04' 
      AND occurred_at::timestamp < '2014-08-11' THEN 'week_0804'
      WHEN occurred_at::timestamp >= '2014-08-11' 
      AND occurred_at::timestamp < '2014-08-18' THEN 'week_0811'
      WHEN occurred_at::timestamp >= '2014-08-18' 
      AND occurred_at::timestamp < '2014-08-25' THEN 'week_0818'
      WHEN occurred_at::timestamp >= '2014-08-25' 
      AND occurred_at::timestamp < '2014-09-01' THEN 'week_0825'
      WHEN occurred_at::timestamp >= '2014-09-01' 
      AND occurred_at::timestamp < '2014-09-08' THEN 'week_0901'
      ELSE 'out of range' END AS interval
FROM tutorial.yammer_events) events
LEFT JOIN tutorial.yammer_users users
ON events.user_id = users.user_id
WHERE state = 'active'
GROUP BY 1,2 ORDER BY 1,2

####### Conclusion:
-- certain mobile devices (eg: iphone 5s, Nexus 5, galaxy S4  ) have significant drop in counts after 07/28 week. This could be due to  few bugs on mobile-application updates.
-- So the drop in active user counts is caused by below 2 major reasons:
-- 1) Users  with more than 3-4 weeks tenure are  not engaged much.
		Plan of action:  Employer should take corrective measure to improve user engagement.
		Investigate, the correlation between user engagement and attrition rate in the organization.
		
-- 2) mobile devices might have had an issue with an application upgrade:
         Plan of action:  test the fore-menstioned devices for any bugs and fix them as necessary.
		 