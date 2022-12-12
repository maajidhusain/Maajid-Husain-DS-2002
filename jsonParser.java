import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.stream.Collectors;

public class jsonParser {
    public static void main(String[] args) throws IOException {
        //read in file
        File file = new File("lab04.json");
        InputStream inputStream = file.;
        InputStreamReader inputStreamReader = new InputStreamReader(inputStream, StandardCharsets.UTF_8);
        BufferedReader bufferedReader = new BufferedReader(inputStreamReader);
        String jsonString = bufferedReader.lines().collect(Collectors.joining());
        bufferedReader.close();
    }
}