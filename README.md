#  Health Evaluation Indicators and Measurement Method for Open-Source Software Projects
(1)create GHTorrent MySql database
First, you need load GHTorrent to your local computer. Step is as follows:
	
	(a)download ghtorrent dataset:
		site: https://www.ghtorrent.org/downloads.html
	
	(b)decompress the dataset：
		tar zxvf mysql-2018-06-01.tar.gz
	
	(c)connect to the mysql database, create a new database and user (you can replace 'ghtorrentuser','ghtorrentpassword' with others):
		mysql -u root -p
		create user 'ghtorrentuser'@'localhost' identified by 'ghtorrentpassword';
		create user 'ghtorrentuser'@'*' identified by 'ghtorrentpassword';
		create database ghtorrent_restore;
		grant all privileges on ghtorrent_restore.* to 'ghtorrentuser'@'localhost';
		grant all privileges on ghtorrent_restore.* to 'ghtorrentuser'@'*';
		grant file on *.* to 'ghtorrentuser'@'localhost';
	
	(d)import the decompressed data into the database ghtorrent_restore:
		cd mysql-2018-06-01
		./ght-restore-mysql -u ghtorrentuser -d ghtorrent_restore -p ghtorrentpassword . 
	
	(e)if you want to visit database remotely, then:
		CREATE USER 'ystian'@'%' IDENTIFIED BY '123456';
		GRANT select ON ghtorrent_restore.* TO 'ystian'@'%'

(2)obtain the experimental data, namely daily data of all projects from created date.
	
	run  src/data/main.py
	
	we sort by project id, you can get the projects in [start*100th, end*100th] by modifying the value of 'start' and 'end'
	we got 45000 projects, shown in data/

(3)get experimental result to answer four key question:
	
	run src/model/main.py 
	
	Question 1: Is the method of health measurement effective?
	Question 2: What is the distribution of OSS project health in an open-source community?
	Question 3: How do the evaluation indicators and project health in different distribution areas change with time?
	Question 4: At different stages, what is the change of the importance of each evaluation indicators to measure the health of OSS project?
	
	experimental result is shown in fig/ .