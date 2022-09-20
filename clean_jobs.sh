find /home/openreview/openreview-expertise/jobs/*/mfr -prune -type d -mtime +0.168 -exec rm -r {} \;
find /home/openreview/openreview-expertise/jobs/*/specter -prune -type d -mtime +0.168 -exec rm -r {} \;
find /home/openreview/openreview-expertise/jobs/*/sub2vec.jsonl -prune -mtime +0.168 -exec rm -r {} \;
