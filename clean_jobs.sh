find /home/openreview/openreview-expertise/jobs/*/archives -prune -type d -mtime +2 -exec rm -r {} \;
find /home/openreview/openreview-expertise/jobs/*/mfr -prune -type d -mtime +2 -exec rm -r {} \;
find /home/openreview/openreview-expertise/jobs/*/specter -prune -type d -mtime +2 -exec rm -r {} \;
find /home/openreview/openreview-expertise/jobs/*/sub2vec.jsonl -prune -mtime +2 -exec rm -r {} \;
find /home/openreview/openreview-expertise/jobs/*/metadata.json -prune -mtime +2 -exec rm -r {} \;
find /home/openreview/openreview-expertise/jobs/*/submissions.json -prune -mtime +2 -exec rm -r {} \;
find /home/openreview/openreview-expertise/jobs/*/pub2vec.jsonl -prune -mtime +2 -exec rm -r {} \;
