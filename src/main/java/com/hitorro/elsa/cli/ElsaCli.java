package com.hitorro.elsa.cli;

import com.hitorro.elsa.*;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Scanner;

/**
 * Interactive CLI for ELSA entity-level sentiment analysis.
 * Paste or type text, get entity-sentiment relationships back.
 */
public class ElsaCli {

    private static final String ANSI_RESET = "\033[0m";
    private static final String ANSI_GREEN = "\033[32m";
    private static final String ANSI_RED = "\033[31m";
    private static final String ANSI_YELLOW = "\033[33m";
    private static final String ANSI_CYAN = "\033[36m";
    private static final String ANSI_BOLD = "\033[1m";
    private static final String ANSI_DIM = "\033[2m";

    public static void main(String[] args) throws Exception {
        Path modelDir = Path.of(args.length > 0 ? args[0] : "models");

        System.out.println(ANSI_BOLD + "ELSA - Entity-Level Sentiment Analysis" + ANSI_RESET);
        System.out.println(ANSI_DIM + "Loading models from: " + modelDir + ANSI_RESET);

        EntitySentimentAnalyzer analyzer = EntitySentimentAnalyzer.create(
                AnalyzerConfig.builder().modelDirectory(modelDir).build());

        System.out.println(ANSI_GREEN + "Ready." + ANSI_RESET);
        System.out.println();
        System.out.println("Paste or type text, then press Enter twice (empty line) to analyze.");
        System.out.println("Type " + ANSI_BOLD + "quit" + ANSI_RESET + " to exit.");
        System.out.println();

        Scanner scanner = new Scanner(System.in);

        while (true) {
            System.out.print(ANSI_CYAN + "elsa> " + ANSI_RESET);
            String text = readInput(scanner);

            if (text == null || text.equalsIgnoreCase("quit") || text.equalsIgnoreCase("exit")) {
                break;
            }
            if (text.isBlank()) {
                continue;
            }

            AnalysisResult result = analyzer.analyze(text);
            printResult(result);
        }

        analyzer.close();
        System.out.println("Bye.");
    }

    private static String readInput(Scanner scanner) {
        StringBuilder sb = new StringBuilder();
        while (scanner.hasNextLine()) {
            String line = scanner.nextLine();
            if (sb.isEmpty() && (line.equalsIgnoreCase("quit") || line.equalsIgnoreCase("exit"))) {
                return line;
            }
            if (line.isEmpty() && !sb.isEmpty()) {
                break; // empty line after content = submit
            }
            if (!line.isEmpty()) {
                if (!sb.isEmpty()) sb.append(' ');
                sb.append(line);
            }
        }
        if (!scanner.hasNextLine() && sb.isEmpty()) {
            return null; // EOF
        }
        return sb.toString();
    }

    private static void printResult(AnalysisResult result) {
        System.out.println();

        if (!result.subjectiveContent()) {
            System.out.println(ANSI_DIM + "  Classified as objective/factual — no sentiment analysis performed." + ANSI_RESET);
            System.out.printf(ANSI_DIM + "  (%d ms)%n" + ANSI_RESET, result.elapsed().toMillis());
            System.out.println();
            return;
        }

        if (!result.hasEntities()) {
            System.out.println(ANSI_DIM + "  No entities found in sentiment-bearing sentences." + ANSI_RESET);
            System.out.printf(ANSI_DIM + "  Sentences analyzed: %d, skipped: %d (%d ms)%n" + ANSI_RESET,
                    result.sentencesAnalyzed(), result.sentencesSkipped(), result.elapsed().toMillis());
            System.out.println();
            return;
        }

        // Header
        System.out.printf("  " + ANSI_BOLD + "%-28s %-8s %-12s %-6s  %s" + ANSI_RESET + "%n",
                "ENTITY", "TYPE", "SENTIMENT", "CONF", "HOLDER");
        System.out.println("  " + "-".repeat(78));

        for (EntitySentiment es : result.entities()) {
            String sentColor = switch (es.sentiment()) {
                case POSITIVE -> ANSI_GREEN;
                case NEGATIVE -> ANSI_RED;
                case MIXED -> ANSI_YELLOW;
                case NEUTRAL -> ANSI_DIM;
            };

            System.out.printf("  %-28s %-8s %s%-12s%s %-6.2f  %s%n",
                    truncate(es.entity(), 28),
                    es.entityType(),
                    sentColor, es.sentiment(), ANSI_RESET,
                    es.confidence(),
                    es.holder() != null ? es.holder() : "");
        }

        System.out.println("  " + "-".repeat(78));
        System.out.printf(ANSI_DIM + "  %d entities | %d sentences analyzed, %d skipped | %d ms%n" + ANSI_RESET,
                result.entities().size(),
                result.sentencesAnalyzed(), result.sentencesSkipped(),
                result.elapsed().toMillis());
        System.out.println();
    }

    private static String truncate(String s, int max) {
        return s.length() <= max ? s : s.substring(0, max - 1) + "\u2026";
    }
}
