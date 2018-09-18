#!/usr/bin/perl -w

use strict;
use utf8;
 
use open ":utf8";
no warnings "utf8";

binmode (STDIN, ":utf8");
binmode (STDOUT, ":utf8");
binmode (STDERR, ":utf8");

die "Not enought arguments!" unless $ARGV[3];
my $corpus_path = $ARGV[0];
my $result_path = $ARGV[1];
my $ws = int($ARGV[2]);
my $vocab_path = $ARGV[3];

open VOC_F, '<', $vocab_path // die "Can' open $vocab_path";
open INP_F, '<', $corpus_path // die "Can't open $corpus_path";
open OUT_F, '>', $result_path // die "Can't open $result_path";

my %ind = ();
my $last_ind = 0;
my @h = ();
my @all_words = ();

while (my $line = <VOC_F>) {
	chomp $line;
	$line =~ s/^\s+|\s+$//g;
	my ($freq, $word) = split(/\s+/, $line);
	if ($freq > 20 && not $ind{$word}) {
		$ind{$word} = $last_ind;
		push @all_words, $word;
		push @h, {};
		$last_ind++;
	}
}
print STDERR "count words: ",scalar(@all_words),"\n";
my $cnt_lines = 0;

while (my $line = <INP_F>){
	$cnt_lines++;
	if ($cnt_lines % 100_000 == 0) {
		print STDERR "$cnt_lines done\n";
	}
    chomp $line;
	my @words = split(/\s+/, $line);
	my $cnt_words = scalar(@words);
	my $wi;
	for (my $i=0; $i<$cnt_words; $i++) {
		for (my $wi = $i-$ws; $wi < $i+$ws; $wi++) {
			next if ($wi < 0 || $wi >= $cnt_words || $wi == $i);

			unless ($ind{$words[$i]}) {
				next;
			}
			my $i1 = $ind{$words[$i]};

			unless ($ind{$words[$wi]}) {
				next;
			}
			my $i2 = $ind{$words[$wi]};

			$h[$i1]->{$i2} += $ws - abs($i - $wi) + 1;
		}
	}
}
close INP_F;

for my $word_1 (keys %ind) {
	my $i1 = $ind{$word_1};
	for my $i2 (keys %{ $h[$i1] } ) {
		print OUT_F join("\t", $word_1, $all_words[$i2], $h[$i1]->{$i2}),"\n";
	}
}
close OUT_F;
