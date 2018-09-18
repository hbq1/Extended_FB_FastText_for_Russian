#!/usr/bin/perl -w

use strict;
use utf8;
 
use open ":utf8";
no warnings "utf8";

binmode (STDIN, ":utf8");
binmode (STDOUT, ":utf8");
binmode (STDERR, ":utf8");

die "Not enought arguments!" unless $ARGV[2];
my $corpus_path = $ARGV[0];
my $word_ngram_path = $ARGV[1];
my $ngram_info_path = $ARGV[2];

open IN_F, '<', $corpus_path // die "Can't open $corpus_path";
open OUT_W_N, '>', $word_ngram_path // die "Can't open $word_ngram_path";
open OUT_N_INFO, '>', $ngram_info_path // die "Can't open $ngram_info_path";

my %h = ();
while (my $line = <IN_F>){
    chomp $line;
	$line =~ s/^\s+|\s+$//g;
	my ($rep, $word_full) = split(/\s+/, $line);
	my @word = split('', "\<$word_full\>");
	my @ngrams = ();
	my $len = scalar(@word);
	for (my $i=0; $i+2 < $len; $i++) {
		my $ngram = join('', map{ $word[$_] } $i..($i+1));
		for(my $j=$i+2; $j<$len && $j<$i+6; $j++) {
			$ngram .= $word[$j];
			push @ngrams, $ngram;
			$h{$ngram}{cnt}++;
			$h{$ngram}{sum} += $rep;
		}
	}
	print OUT_W_N "$word_full\t".join(' ', @ngrams)."\n";
}
close IN_F;
close OUT_W_N;

for my $word(keys %h) {
	print OUT_N_INFO "$word\t$h{$word}{cnt}\t$h{$word}{sum}\n";
}

close OUT_N_INFO;

