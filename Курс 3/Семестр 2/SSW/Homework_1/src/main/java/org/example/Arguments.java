package org.example;

import java.util.List;

public class Arguments {
    private final List<String> inputFiles;
    private final String outputPath;
    private final String prefix;
    private final boolean appendEnabled;
    private final StatsMode statsMode;

    public Arguments(List<String> inputFiles, String outputPath, String prefix, boolean appendEnabled, StatsMode statsMode) {
        this.inputFiles = inputFiles;
        this.outputPath = outputPath;
        this.prefix = prefix;
        this.appendEnabled = appendEnabled;
        this.statsMode = statsMode;
    }

    public List<String> getInputFiles() { return inputFiles; }
    public String getOutputPath() { return outputPath; }
    public String getPrefix() { return prefix; }
    public boolean isAppendEnabled() { return appendEnabled; }
    public StatsMode getStatsMode() { return statsMode; }
}