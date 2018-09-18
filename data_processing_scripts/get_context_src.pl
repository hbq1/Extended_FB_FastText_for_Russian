#!/usr/bin/perl -w

use strict;
use utf8;
 
use open ":utf8";
no warnings "utf8";

binmode (STDIN, ":utf8");
binmode (STDOUT, ":utf8");
binmode (STDERR, ":utf8");

die "Not enought arguments!" unless $ARGV[3];
my $contexts_path = $ARGV[0];
my $vocab_path = $ARGV[1];
my $word_context_path = $ARGV[2];
my $context_info_path = $ARGV[3];

open IN_F, '<', $contexts_path // die "Can't open $contexts_path";
open VOCAB_F, '<', $vocab_path // die "Can't open $vocab_path";
open OUT_W_C, '>', $word_context_path // die "Can't open $word_context_path";
open OUT_C_INFO, '>', $context_info_path // die "Can't open $context_info_path";
binmode VOCAB_F, 'utf8';


my %vocab = ();
while (my $line = <VOCAB_F>) {
	chomp $line;
	$line =~ s/^\s+|\s+$//g;
	my ($freq, $word) = split(/\s+/, $line);
	$vocab{$word} = $freq;
}
close VOCAB_F;


my %h = ();
my $prev_w = '';
my @buf = ();
my $cnt = 0;
while (my $line = <IN_F>){
    chomp $line;
	$line =~ s/^\s+|\s+$//g;
	my ($w1, $w2, $score) = split(/\s+/, $line);
	next unless ($vocab{$w1} && $vocab{$w2});
	if ($w1 ne $prev_w) {
		if ( scalar(@buf) > 0 ) {
			print OUT_W_C $prev_w."\t".join(' ', @buf),"\n";
		}
		$prev_w = $w1;
		$cnt = 1;
		@buf = ($w2);
		$h{$w2}{cnt}++;
		$h{$w2}{sum} += $vocab{$w1};
	} else {
		if ($cnt < 50) {
			push @buf, $w2;
			$h{$w2}{cnt}++;
			$h{$w2}{sum} += $vocab{$w1};
		}
		$cnt++;
	}
}
if ( scalar(@buf) > 0 ) {
	print OUT_W_C $prev_w."\t".join(' ', @buf),"\n";
}

close IN_F;
close OUT_W_C;

for my $word(keys %h) {
	print OUT_C_INFO "$word\t$h{$word}{cnt}\t$h{$word}{sum}\n";
}

close OUT_C_INFO;

