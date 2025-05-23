import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class SentimentMapper extends Mapper<Object, Text, Text, Text> {
    private final Map<String, Integer> afinnScores = new HashMap<>();

    @Override
    protected void setup(Context context) throws IOException {
        // Load the AFINN-111.txt file from the distributed cache
        BufferedReader br = new BufferedReader(new FileReader("AFINN-111.txt"));
        String line;
        while ((line = br.readLine()) != null) {
            String[] parts = line.split("\t");
            if (parts.length == 2) {
                afinnScores.put(parts[0], Integer.parseInt(parts[1]));
            }
        }
        br.close();
    }

    @Override
    protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        String[] fields = value.toString().split(",", -1);
        if (fields.length > 2) {
            String commentID = fields[2].trim();  // Use comment_id (3rd column)
            String commentText = fields[1].trim();  // Use comment_text (2nd column)
            int score = 0;

            // Perform sentiment analysis on the comment text
            for (String word : commentText.split("\\s+")) {
                score += afinnScores.getOrDefault(word.replaceAll("[^a-zA-Z]", ""), 0);
            }

            // Determine sentiment based on score
            String sentiment = "Neutral";
            if (score > 0) {
                sentiment = "Positive";
            } else if (score < 0) {
                sentiment = "Negative";
            }

            // Emit only comment_id and sentiment (omit comment_text)
            context.write(new Text(commentID), new Text(sentiment));
        }
    }
}

